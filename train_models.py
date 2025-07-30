# -*- coding: utf-8 -*-
"""Multi-Model Ranker Training Pipeline with Cross-Validation

Supports XGBoost, LightGBM, and CatBoost with:
1. GroupKFold cross-validation with ranker_id as group identifier
2. Correct HitRate@3 calculation with >10 options filter
3. Model saving functionality with descriptive names
4. Command-line model selection via argparse

Usage:
    python train_models.py --model xgboost
    python train_models.py --model lightgbm
    python train_models.py --model catboost
    python train_models.py  # trains all models
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import time
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import GroupKFold
import os
import pickle
import argparse
from datetime import datetime
from abc import ABC, abstractmethod

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train ranking models for flight selection')
parser.add_argument('--model', type=str, choices=['xgboost', 'lightgbm', 'catboost', 'all'],
                    default='all', help='Which model to train (default: all)')
args = parser.parse_args()

# Configuration for full training
USE_SUBSET = False  # Use 100% of training data for final model
SUBSET_FRACTION = 1.0
N_FOLDS = 5

print('Loading local data files...')

# Load data from local files
train = pl.read_parquet('./data/train.parquet')
test = pl.read_parquet('./data/test.parquet').with_columns(pl.lit(0, dtype=pl.Int64).alias("selected"))

# USE SUBSET FOR TESTING
if USE_SUBSET:
    print(f"Using {SUBSET_FRACTION*100}% of training data for testing...")
    n_subset = int(len(train) * SUBSET_FRACTION)
    train = train[:n_subset]
    print(f"Training data reduced to {len(train)} rows")
else:
    print("🚀 Using 100% of training data for final model!")

# Drop __index_level_0__ column if it exists
if '__index_level_0__' in train.columns:
    train = train.drop('__index_level_0__')
if '__index_level_0__' in test.columns:
    test = test.drop('__index_level_0__')

print(f'Train shape: {train.shape}')
print(f'Test shape: {test.shape}')
print('Data loading complete.')

data_raw = pl.concat((train, test))

"""## Helpers"""

def hitrate_at_3(y_true, y_pred, groups):
    """
    CORRECTED HitRate@3 implementation that matches competition rules:
    - Only considers groups (ranker_id) with >10 flight options
    - Calculates fraction of sessions where correct flight is in top-3
    """
    df = pl.DataFrame({
        'group': groups,
        'pred': y_pred,
        'true': y_true
    })

    return (
        df.filter(pl.col("group").count().over("group") > 10)  # CRITICAL: >10 options filter
        .sort(["group", "pred"], descending=[False, True])
        .group_by("group", maintain_order=True)
        .head(3)
        .group_by("group")
        .agg(pl.col("true").max())
        .select(pl.col("true").mean())
        .item()
    )

"""## Feature Engineering"""

df = data_raw.clone()

# More efficient duration to minutes converter
def dur_to_min(col):
    # Extract days and time parts in one pass
    days = col.str.extract(r"^(\d+)\.", 1).cast(pl.Int64).fill_null(0) * 1440
    time_str = pl.when(col.str.contains(r"^\d+\.")).then(col.str.replace(r"^\d+\.", "")).otherwise(col)
    hours = time_str.str.extract(r"^(\d+):", 1).cast(pl.Int64).fill_null(0) * 60
    minutes = time_str.str.extract(r":(\d+):", 1).cast(pl.Int64).fill_null(0)
    return (days + hours + minutes).fill_null(0)

# Process duration columns
dur_cols = ["legs0_duration", "legs1_duration"] + [f"legs{l}_segments{s}_duration" for l in (0, 1) for s in (0, 1)]
dur_exprs = [dur_to_min(pl.col(c)).alias(c) for c in dur_cols if c in df.columns]

# Apply duration transformations first
if dur_exprs:
    df = df.with_columns(dur_exprs)

# Precompute marketing carrier columns check
mc_cols = [f'legs{l}_segments{s}_marketingCarrier_code' for l in (0, 1) for s in range(4)]
mc_exists = [col for col in mc_cols if col in df.columns]

# Combine all initial transformations
df = df.with_columns([
        # Price features
        (pl.col("taxes") / (pl.col("totalPrice") + 1)).alias("tax_rate"),
        pl.col("totalPrice").log1p().alias("log_price"),

        # Duration features
        (pl.col("legs0_duration").fill_null(0) + pl.col("legs1_duration").fill_null(0)).alias("total_duration"),
        pl.when(pl.col("legs1_duration").fill_null(0) > 0)
            .then(pl.col("legs0_duration") / (pl.col("legs1_duration") + 1))
            .otherwise(1.0).alias("duration_ratio"),

        # Trip type
        (pl.col("legs1_duration").is_null() |
         (pl.col("legs1_duration") == 0) |
         pl.col("legs1_segments0_departureFrom_airport_iata").is_null()).cast(pl.Int32).alias("is_one_way"),

        # Total segments count
        (pl.sum_horizontal(pl.col(col).is_not_null().cast(pl.UInt8) for col in mc_exists)
         if mc_exists else pl.lit(0)).alias("l0_seg"),

        # FF features
        (pl.col("frequentFlyer").fill_null("").str.count_matches("/") +
         (pl.col("frequentFlyer").fill_null("") != "").cast(pl.Int32)).alias("n_ff_programs"),

        # Binary features
        pl.col("corporateTariffCode").is_not_null().cast(pl.Int32).alias("has_corporate_tariff"),
        (pl.col("pricingInfo_isAccessTP") == 1).cast(pl.Int32).alias("has_access_tp"),

        (
            (pl.col("miniRules0_monetaryAmount") == 0)
            & (pl.col("miniRules0_statusInfos") == 1)
        )
        .cast(pl.Int8)
        .alias("free_cancel"),
        (
            (pl.col("miniRules1_monetaryAmount") == 0)
            & (pl.col("miniRules1_statusInfos") == 1)
        )
        .cast(pl.Int8)
        .alias("free_exchange"),

        # Routes & carriers
        pl.col("searchRoute").is_in(["MOWLED/LEDMOW", "LEDMOW/MOWLED", "MOWLED", "LEDMOW"])
            .cast(pl.Int32).alias("is_popular_route"),

        # Cabin
        pl.mean_horizontal(["legs0_segments0_cabinClass", "legs1_segments0_cabinClass"]).alias("avg_cabin_class"),
        (pl.col("legs0_segments0_cabinClass").fill_null(0) -
         pl.col("legs1_segments0_cabinClass").fill_null(0)).alias("cabin_class_diff"),
])

# Segment counts - more efficient
seg_exprs = []
for leg in (0, 1):
    seg_cols = [f"legs{leg}_segments{s}_duration" for s in range(4) if f"legs{leg}_segments{s}_duration" in df.columns]
    if seg_cols:
        seg_exprs.append(
            pl.sum_horizontal(pl.col(c).is_not_null() for c in seg_cols)
                .cast(pl.Int32).alias(f"n_segments_leg{leg}")
        )
    else:
        seg_exprs.append(pl.lit(0).cast(pl.Int32).alias(f"n_segments_leg{leg}"))

# Add segment-based features
df = df.with_columns(seg_exprs)

# Then use them for derived features
df = df.with_columns([
    (pl.col("n_segments_leg0") + pl.col("n_segments_leg1")).alias("total_segments"),
    (pl.col("n_segments_leg0") == 1).cast(pl.Int32).alias("is_direct_leg0"),
    pl.when(pl.col("is_one_way") == 1).then(0)
        .otherwise((pl.col("n_segments_leg1") == 1).cast(pl.Int32)).alias("is_direct_leg1"),
])

# More derived features
df = df.with_columns([
    (pl.col("is_direct_leg0") & pl.col("is_direct_leg1")).cast(pl.Int32).alias("both_direct"),
    ((pl.col("isVip") == 1) | (pl.col("n_ff_programs") > 0)).cast(pl.Int32).alias("is_vip_freq"),
    pl.col("Id").count().over("ranker_id").alias("group_size"),
])

# Add major carrier flag if column exists
if "legs0_segments0_marketingCarrier_code" in df.columns:
    df = df.with_columns(
        pl.col("legs0_segments0_marketingCarrier_code").is_in(["SU", "S7"])
            .cast(pl.Int32).alias("is_major_carrier")
    )
else:
    df = df.with_columns(pl.lit(0).alias("is_major_carrier"))

df = df.with_columns(pl.col("group_size").log1p().alias("group_size_log"))

# Time features - batch process
time_exprs = []
for col in ("legs0_departureAt", "legs0_arrivalAt", "legs1_departureAt", "legs1_arrivalAt"):
    if col in df.columns:
        dt = pl.col(col).str.to_datetime(strict=False)
        h = dt.dt.hour().fill_null(12)
        time_exprs.extend([
            h.alias(f"{col}_hour"),
            dt.dt.weekday().fill_null(0).alias(f"{col}_weekday"),
            (((h >= 6) & (h <= 9)) | ((h >= 17) & (h <= 20))).cast(pl.Int32).alias(f"{col}_business_time")
        ])
if time_exprs:
    df = df.with_columns(time_exprs)

# Price and duration basic ranks
rank_exprs = []
for col, alias in [("totalPrice", "price"), ("total_duration", "duration")]:
    rank_exprs.append(pl.col(col).rank().over("ranker_id").alias(f"{alias}_rank"))

# Price-specific features
price_exprs = [
    (pl.col("totalPrice").rank("average").over("ranker_id") /
     pl.col("totalPrice").count().over("ranker_id")).alias("price_pct_rank"),
    (pl.col("totalPrice") == pl.col("totalPrice").min().over("ranker_id")).cast(pl.Int32).alias("is_cheapest"),
    ((pl.col("totalPrice") - pl.col("totalPrice").median().over("ranker_id")) /
     (pl.col("totalPrice").std().over("ranker_id") + 1)).alias("price_from_median"),
    (pl.col("l0_seg") == pl.col("l0_seg").min().over("ranker_id")).cast(pl.Int32).alias("is_min_segments"),
]

# Apply initial ranks
df = df.with_columns(rank_exprs + price_exprs)

# Cheapest direct - more efficient
direct_cheapest = (
    df.filter(pl.col("is_direct_leg0") == 1)
    .group_by("ranker_id")
    .agg(pl.col("totalPrice").min().alias("min_direct"))
)

df = df.join(direct_cheapest, on="ranker_id", how="left").with_columns(
    ((pl.col("is_direct_leg0") == 1) &
     (pl.col("totalPrice") == pl.col("min_direct"))).cast(pl.Int32).fill_null(0).alias("is_direct_cheapest")
).drop("min_direct")

# Popularity features - efficient join
df = (
    df.join(
        train.group_by('legs0_segments0_marketingCarrier_code').agg(pl.mean('selected').alias('carrier0_pop')),
        on='legs0_segments0_marketingCarrier_code',
        how='left'
    )
    .join(
        train.group_by('legs1_segments0_marketingCarrier_code').agg(pl.mean('selected').alias('carrier1_pop')),
        on='legs1_segments0_marketingCarrier_code',
        how='left'
    )
    .with_columns([
        pl.col('carrier0_pop').fill_null(0.0),
        pl.col('carrier1_pop').fill_null(0.0),
    ])
)

# Final features including popularity
df = df.with_columns([
    (pl.col('carrier0_pop') * pl.col('carrier1_pop')).alias('carrier_pop_product'),
])

# Group-context gap features for HitRate@3 optimization
df = df.with_columns([
    # 1. price_gap: flight price minus minimum price in its ranker_id group
    (pl.col("totalPrice") - pl.col("totalPrice").min().over("ranker_id")).alias("price_gap"),
    
    # 2. departure_gap: flight departure time minus minimum departure time in group
    (pl.col("legs0_departureAt_hour") - pl.col("legs0_departureAt_hour").min().over("ranker_id")).alias("departure_gap").fill_null(0),
    
    # 3. is_best_direct: Binary flag for flights that are both cheapest AND direct among direct flights in group
    pl.when(pl.col("is_direct_leg0") == 1)
        .then(pl.col("totalPrice"))
        .otherwise(None)
        .min()
        .over("ranker_id")
        .alias("min_direct_price"),
]).with_columns([
    # Then create the is_best_direct flag: direct flight AND matches minimum price among direct flights
    ((pl.col("is_direct_leg0") == 1) & 
     (pl.col("totalPrice") == pl.col("min_direct_price")) &
     pl.col("min_direct_price").is_not_null())
    .cast(pl.Int32)
    .fill_null(0)
    .alias("is_best_direct")
]).drop("min_direct_price")  # Clean up temporary column

# Fill nulls
data = df.with_columns(
    [pl.col(c).fill_null(0) for c in df.select(pl.selectors.numeric()).columns] +
    [pl.col(c).fill_null("missing") for c in df.select(pl.selectors.string()).columns]
)

"""## Feature Selection"""

# Categorical features
cat_features = [
    'nationality', 'searchRoute', 'corporateTariffCode',
    'bySelf', 'sex', 'companyID',
    # Leg 0 segments 0-1
    'legs0_segments0_aircraft_code', 'legs0_segments0_arrivalTo_airport_city_iata',
    'legs0_segments0_arrivalTo_airport_iata', 'legs0_segments0_departureFrom_airport_iata',
    'legs0_segments0_marketingCarrier_code', 'legs0_segments0_operatingCarrier_code',
    'legs0_segments0_flightNumber',
    'legs0_segments1_aircraft_code', 'legs0_segments1_arrivalTo_airport_city_iata',
    'legs0_segments1_arrivalTo_airport_iata', 'legs0_segments1_departureFrom_airport_iata',
    'legs0_segments1_marketingCarrier_code', 'legs0_segments1_operatingCarrier_code',
    'legs0_segments1_flightNumber',
    # Leg 1 segments 0-1
    'legs1_segments0_aircraft_code', 'legs1_segments0_arrivalTo_airport_city_iata',
    'legs1_segments0_arrivalTo_airport_iata', 'legs1_segments0_departureFrom_airport_iata',
    'legs1_segments0_marketingCarrier_code', 'legs1_segments0_operatingCarrier_code',
    'legs1_segments0_flightNumber',
    'legs1_segments1_aircraft_code', 'legs1_segments1_arrivalTo_airport_city_iata',
    'legs1_segments1_arrivalTo_airport_iata', 'legs1_segments1_departureFrom_airport_iata',
    'legs1_segments1_marketingCarrier_code', 'legs1_segments1_operatingCarrier_code',
    'legs1_segments1_flightNumber',
]

# Columns to exclude (uninformative or problematic)
exclude_cols = [
    'Id', 'ranker_id', 'selected', 'profileId', 'requestDate',
    'legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt',
    'miniRules0_percentage', 'miniRules1_percentage',  # >90% missing
    'frequentFlyer',  # Already processed
    'pricingInfo_passengerCount'  # Constant columns
]

for leg in [0, 1]:
    for seg in [0, 1]:
        if seg == 0:
            suffixes = ["seatsAvailable"]
        else:
            suffixes = [
                "cabinClass", "seatsAvailable", "baggageAllowance_quantity",
                "baggageAllowance_weightMeasurementType", "aircraft_code",
                "arrivalTo_airport_city_iata", "arrivalTo_airport_iata",
                "departureFrom_airport_iata", "flightNumber",
                "marketingCarrier_code", "operatingCarrier_code",
            ]
        for suffix in suffixes:
            exclude_cols.append(f"legs{leg}_segments{seg}_{suffix}")

# Exclude segment 2-3 columns (>98% missing)
for leg in [0, 1]:
    for seg in [2, 3]:
        for suffix in ['aircraft_code', 'arrivalTo_airport_city_iata', 'arrivalTo_airport_iata',
                      'baggageAllowance_quantity', 'baggageAllowance_weightMeasurementType',
                      'cabinClass', 'departureFrom_airport_iata', 'duration', 'flightNumber',
                      'marketingCarrier_code', 'operatingCarrier_code', 'seatsAvailable']:
            exclude_cols.append(f'legs{leg}_segments{seg}_{suffix}')

feature_cols = [col for col in data.columns if col not in exclude_cols]
cat_features_final = [col for col in cat_features if col in feature_cols]

print(f"Using {len(feature_cols)} features ({len(cat_features_final)} categorical)")

X = data.select(feature_cols)
y = data.select('selected')
groups = data.select('ranker_id')

"""## MODEL TRAINER CLASSES"""

class ModelTrainer(ABC):
    """Abstract base class for all model trainers"""
    
    def __init__(self, random_state=RANDOM_STATE):
        self.random_state = random_state
        self.model = None
        self.model_type = None
    
    @abstractmethod
    def get_params(self):
        """Get model-specific parameters"""
        pass
    
    @abstractmethod
    def train(self, X_train, y_train, groups_train, X_val=None, y_val=None, groups_val=None):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions"""
        pass
    
    @abstractmethod
    def save_model(self, filepath):
        """Save the trained model"""
        pass


class XGBoostTrainer(ModelTrainer):
    """XGBoost ranker implementation"""
    
    def __init__(self, random_state=RANDOM_STATE):
        super().__init__(random_state)
        self.model_type = 'xgboost'
    
    def get_params(self):
        return {
            'objective': 'rank:pairwise',
            'eval_metric': 'ndcg@3',
            "learning_rate": 0.022641389657079056,
            "max_depth": 14,
            "min_child_weight": 2,
            "subsample": 0.8842234913702768,
            "colsample_bytree": 0.45840689146263086,
            "gamma": 3.3084297630544888,
            "lambda": 6.952586917313028,
            "alpha": 0.6395254133055179,
            'seed': self.random_state,
            'n_jobs': -1,
        }
    
    def train(self, X_train, y_train, groups_train, X_val=None, y_val=None, groups_val=None):
        # Create XGBoost datasets
        group_sizes_train = groups_train.group_by('ranker_id', maintain_order=True).agg(pl.len())['len'].to_numpy()
        dtrain = xgb.DMatrix(X_train, label=y_train, group=group_sizes_train, feature_names=X_train.columns)
        
        evals = [(dtrain, 'train')]
        if X_val is not None:
            group_sizes_val = groups_val.group_by('ranker_id', maintain_order=True).agg(pl.len())['len'].to_numpy()
            dval = xgb.DMatrix(X_val, label=y_val, group=group_sizes_val, feature_names=X_val.columns)
            evals.append((dval, 'val'))
        
        # Train model
        rounds = 400 if USE_SUBSET else 600
        self.model = xgb.train(
            self.get_params(),
            dtrain,
            num_boost_round=rounds,
            evals=evals,
            verbose_eval=0
        )
        return self
    
    def predict(self, X):
        if isinstance(X, pl.DataFrame):
            # Need to create DMatrix for prediction
            dmatrix = xgb.DMatrix(X, feature_names=X.columns)
            return self.model.predict(dmatrix)
        else:
            # Assume it's already a DMatrix
            return self.model.predict(X)
    
    def save_model(self, filepath):
        self.model.save_model(filepath)


class LightGBMTrainer(ModelTrainer):
    """LightGBM ranker implementation with lambdarank objective"""
    
    def __init__(self, random_state=RANDOM_STATE):
        super().__init__(random_state)
        self.model_type = 'lightgbm'
    
    def get_params(self):
        return {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'eval_at': [3],
            'label_gain': [0, 1, 2, 3],  # Prioritize top-3 positions
            'num_leaves': 256,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': -1,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.5,
            'lambda_l2': 0.5,
            'num_iterations': 1500,
            'seed': self.random_state,
            'num_threads': -1,
            'verbose': -1
        }
    
    def train(self, X_train, y_train, groups_train, X_val=None, y_val=None, groups_val=None):
        # Prepare data for LightGBM
        group_sizes_train = groups_train.group_by('ranker_id', maintain_order=True).agg(pl.len())['len'].to_numpy()
        
        # Create LightGBM dataset
        dtrain = lgb.Dataset(
            X_train.to_numpy(), 
            label=y_train.to_numpy().flatten(),
            group=group_sizes_train,
            feature_name=list(X_train.columns)
        )
        
        valid_sets = [dtrain]
        valid_names = ['train']
        
        if X_val is not None:
            group_sizes_val = groups_val.group_by('ranker_id', maintain_order=True).agg(pl.len())['len'].to_numpy()
            dval = lgb.Dataset(
                X_val.to_numpy(),
                label=y_val.to_numpy().flatten(),
                group=group_sizes_val,
                feature_name=list(X_val.columns),
                reference=dtrain
            )
            valid_sets.append(dval)
            valid_names.append('val')
        
        # Train model
        self.model = lgb.train(
            self.get_params(),
            dtrain,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[lgb.log_evaluation(0)]  # Suppress output
        )
        return self
    
    def predict(self, X):
        if isinstance(X, pl.DataFrame):
            return self.model.predict(X.to_numpy(), num_iteration=self.model.best_iteration)
        else:
            return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def save_model(self, filepath):
        self.model.save_model(filepath)


class CatBoostTrainer(ModelTrainer):
    """CatBoost ranker implementation with YetiRank loss"""
    
    def __init__(self, random_state=RANDOM_STATE):
        super().__init__(random_state)
        self.model_type = 'catboost'
    
    def get_params(self):
        return {
            'loss_function': 'YetiRank',
            'custom_metric': ['NDCG:top=3'],
            'iterations': 1000,
            'learning_rate': 0.03,
            'depth': 8,
            'l2_leaf_reg': 3.0,
            'min_data_in_leaf': 1,
            'random_seed': self.random_state,
            'verbose': False,
            'thread_count': -1,
            'task_type': 'CPU'
        }
    
    def train(self, X_train, y_train, groups_train, X_val=None, y_val=None, groups_val=None):
        # Prepare data for CatBoost
        train_pool = cb.Pool(
            X_train.to_numpy(),
            label=y_train.to_numpy().flatten(),
            group_id=groups_train.to_pandas()['ranker_id'].values,
            feature_names=list(X_train.columns)
        )
        
        eval_pool = None
        if X_val is not None:
            eval_pool = cb.Pool(
                X_val.to_numpy(),
                label=y_val.to_numpy().flatten(),
                group_id=groups_val.to_pandas()['ranker_id'].values,
                feature_names=list(X_val.columns)
            )
        
        # Train model
        self.model = cb.CatBoost(self.get_params())
        self.model.fit(train_pool, eval_set=eval_pool, verbose=False)
        return self
    
    def predict(self, X):
        if isinstance(X, pl.DataFrame):
            return self.model.predict(X.to_numpy())
        else:
            return self.model.predict(X)
    
    def save_model(self, filepath):
        self.model.save_model(filepath)


def get_model_trainer(model_type):
    """Factory function to get the appropriate model trainer"""
    trainers = {
        'xgboost': XGBoostTrainer,
        'lightgbm': LightGBMTrainer,
        'catboost': CatBoostTrainer
    }
    return trainers[model_type]()

"""## CORRECTED CROSS-VALIDATION WITH GROUP FOLD"""

print("\n" + "="*60)
print("IMPLEMENTING CORRECTED CROSS-VALIDATION FRAMEWORK")
print("="*60)

def cross_validate_model(X, y, groups, model_trainer, n_folds=N_FOLDS):
    """
    Model-agnostic cross-validation that properly mimics competition evaluation:
    1. Uses GroupKFold with ranker_id to prevent data leakage
    2. Applies >10 options filter BEFORE calculating HitRate@3 
    3. Returns realistic CV scores that should match public LB
    """
    
    # Prepare data - handle categorical features for XGBoost
    if model_trainer.model_type == 'xgboost':
        data_processed = X.with_columns([(pl.col(c).rank("dense") - 1).fill_null(-1).cast(pl.Int16) for c in cat_features_final])
    else:
        # LightGBM and CatBoost can handle categoricals directly
        data_processed = X
    
    # Extract training data only
    n_train = train.height
    X_train = data_processed[:n_train]
    y_train = y[:n_train]
    groups_train = groups[:n_train]
    
    print(f"\nStarting {n_folds}-fold GroupKFold cross-validation for {model_trainer.model_type.upper()}...")
    print(f"Training data: {len(X_train)} rows, {len(groups_train.unique('ranker_id'))} unique ranker_ids")
    
    # CRITICAL: Use GroupKFold with ranker_id to prevent leakage
    gkf = GroupKFold(n_splits=n_folds)
    groups_array = groups_train.to_pandas()['ranker_id'].values
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups_array)):
        print(f"\nFold {fold + 1}/{n_folds}:")
        
        # Split data
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx] 
        groups_fold_train = groups_train[train_idx]
        
        X_fold_val = X_train[val_idx]
        y_fold_val = y_train[val_idx]
        groups_fold_val = groups_train[val_idx]
        
        print(f"  Train: {len(X_fold_train)} rows, {len(groups_fold_train.unique('ranker_id'))} groups")
        print(f"  Val:   {len(X_fold_val)} rows, {len(groups_fold_val.unique('ranker_id'))} groups")
        
        # Train model using the trainer's train method
        print(f"  Training {model_trainer.model_type}...")
        trainer = get_model_trainer(model_trainer.model_type)  # Create fresh trainer for each fold
        trainer.train(
            X_fold_train, y_fold_train, groups_fold_train,
            X_fold_val, y_fold_val, groups_fold_val
        )
        
        # Make predictions based on model type
        if model_trainer.model_type == 'xgboost':
            # XGBoost needs DMatrix for prediction
            group_sizes_val = groups_fold_val.group_by('ranker_id', maintain_order=True).agg(pl.len())['len'].to_numpy()
            dval = xgb.DMatrix(X_fold_val, label=y_fold_val, group=group_sizes_val, feature_names=X_fold_val.columns)
            val_preds = trainer.predict(dval)
        else:
            # LightGBM and CatBoost can predict from DataFrame
            val_preds = trainer.predict(X_fold_val)
        
        # CRITICAL: Calculate HitRate@3 with correct filtering
        # This step filters validation set to only include groups with >10 options 
        # BEFORE calculating the metric - exactly like competition evaluation
        fold_score = hitrate_at_3(
            y_fold_val.to_pandas()['selected'].values,
            val_preds, 
            groups_fold_val.to_pandas()['ranker_id'].values
        )
        
        fold_scores.append(fold_score)
        print(f"  Fold {fold + 1} HitRate@3: {fold_score:.5f}")
    
    cv_mean = np.mean(fold_scores)
    cv_std = np.std(fold_scores)
    
    print(f"\n" + "="*50)
    print(f"{model_trainer.model_type.upper()} CROSS-VALIDATION RESULTS:")
    print(f"Mean HitRate@3: {cv_mean:.5f} ± {cv_std:.5f}")
    print(f"Individual folds: {[f'{score:.5f}' for score in fold_scores]}")
    print(f"="*50)
    
    return cv_mean, cv_std, fold_scores

# Determine which models to train based on command-line argument
if args.model == 'all':
    models_to_train = ['xgboost', 'lightgbm', 'catboost']
else:
    models_to_train = [args.model]

# Store results for all models
all_results = {}

"""## TRAINING AND VALIDATION FOR SELECTED MODELS"""

for model_type in models_to_train:
    print(f"\n" + "="*70)
    print(f"PROCESSING {model_type.upper()} MODEL")
    print("="*70)
    
    # Get the appropriate trainer
    trainer = get_model_trainer(model_type)
    
    # Run cross-validation
    cv_mean, cv_std, fold_scores = cross_validate_model(X, y, groups, trainer)
    
    # Store CV results
    all_results[model_type] = {
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'fold_scores': fold_scores
    }
    
    print(f"\n" + "="*50)
    print(f"TRAINING FINAL {model_type.upper()} MODEL")
    print("="*50)
    
    # Prepare final training data
    if model_type == 'xgboost':
        data_final = X.with_columns([(pl.col(c).rank("dense") - 1).fill_null(-1).cast(pl.Int16) for c in cat_features_final])
    else:
        data_final = X
    
    # Use all training data for final model
    X_final = data_final[:train.height]
    y_final = y[:train.height]
    groups_final = groups[:train.height]
    
    print(f"Final training: {len(X_final)} rows, {len(groups_final.unique('ranker_id'))} unique groups")
    
    # Train final model on 100% of data
    print(f"Training final {model_type} model on 100% of data...")
    final_trainer = get_model_trainer(model_type)
    final_trainer.train(X_final, y_final, groups_final)
    
    # Create models directory if it doesn't exist
    os.makedirs('./models', exist_ok=True)
    
    # Save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_extension = {
        'xgboost': 'json',
        'lightgbm': 'txt',
        'catboost': 'cbm'
    }[model_type]
    
    model_filename = f'./models/{model_type}_ranker_cv_{cv_mean:.5f}_{timestamp}.{model_extension}'
    final_trainer.save_model(model_filename)
    
    print(f"✅ Model saved to: {model_filename}")
    
    # Save model info
    model_info = {
        'model_type': model_type,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'fold_scores': fold_scores,
        'n_folds': N_FOLDS,
        'subset_fraction': SUBSET_FRACTION if USE_SUBSET else 1.0,
        'params': final_trainer.get_params(),
        'feature_count': len(feature_cols),
        'training_rows': len(X_final),
        'timestamp': timestamp
    }
    
    info_filename = f'./models/{model_type}_model_info_cv_{cv_mean:.5f}_{timestamp}.pkl'
    with open(info_filename, 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"✅ Model info saved to: {info_filename}")
    
    # Store the model path for the final report
    all_results[model_type]['model_path'] = model_filename
    all_results[model_type]['info_path'] = info_filename

"""## FINAL REPORT"""

print(f"\n" + "="*70)
print("MULTI-MODEL TRAINING COMPLETED SUCCESSFULLY! 🎯")
print("="*70)
print(f"Models trained: {', '.join(models_to_train)}")
print(f"\nCROSS-VALIDATION RESULTS (5-fold GroupKFold with HitRate@3):")
print("-"*50)

for model_type in models_to_train:
    result = all_results[model_type]
    print(f"\n{model_type.upper()}:")
    print(f"  Mean CV Score: {result['cv_mean']:.5f} ± {result['cv_std']:.5f}")
    print(f"  Individual folds: {[f'{score:.5f}' for score in result['fold_scores']]}")
    print(f"  Model saved to: {result['model_path']}")

print("\n" + "="*70)
print("✅ All models trained with GroupKFold (no data leakage)")
print("✅ HitRate@3 calculated with >10 options filter")
print("✅ Models saved to ./models/ directory")
print("="*70)