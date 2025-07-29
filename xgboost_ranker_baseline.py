# XGBoost Ranker Baseline - FlightRank 2025
# Based on Kirill's CatBoost ranking baseline adaptation
# Reference: https://www.kaggle.com/code/ka1242/catboost-ranker-baseline-flightrank-2025

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import xgboost as xgb

# Global parameters
TRAIN_SAMPLE_FRAC = 0.2  # Sample 20% of data for faster iteration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Load parquet files
train = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/train.parquet')
test = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/test.parquet')

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"Unique ranker_ids in train: {train['ranker_id'].nunique():,}")
print(f"Selected rate: {train['selected'].mean():.3f}")

# Sample by ranker_id to keep groups intact
if TRAIN_SAMPLE_FRAC < 1.0:
    unique_rankers = train['ranker_id'].unique()
    n_sample = int(len(unique_rankers) * TRAIN_SAMPLE_FRAC)
    sampled_rankers = np.random.RandomState(RANDOM_STATE).choice(
        unique_rankers, size=n_sample, replace=False
    )
    train = train[train['ranker_id'].isin(sampled_rankers)]
    print(f"Sampled train to {len(train):,} rows ({train['ranker_id'].nunique():,} groups)")

# Convert ranker_id to string for consistency
train['ranker_id'] = train['ranker_id'].astype(str)
test['ranker_id'] = test['ranker_id'].astype(str)

# Define categorical features
cat_features = [
    'nationality', 'searchRoute', 'corporateTariffCode',
    # Leg 0 segments 0-1
    'legs0_segments0_aircraft_code', 'legs0_segments0_arrivalTo_airport_city_iata',
    'legs0_segments0_arrivalTo_airport_iata', 'legs0_segments0_departureFrom_airport_iata',
    'legs0_segments0_marketingCarrier_code', 'legs0_segments0_operatingCarrier_code',
    'legs0_segments1_aircraft_code', 'legs0_segments1_arrivalTo_airport_city_iata',
    'legs0_segments1_arrivalTo_airport_iata', 'legs0_segments1_departureFrom_airport_iata',
    'legs0_segments1_marketingCarrier_code', 'legs0_segments1_operatingCarrier_code',
    # Leg 1 segments 0-1
    'legs1_segments0_aircraft_code', 'legs1_segments0_arrivalTo_airport_city_iata',
    'legs1_segments0_arrivalTo_airport_iata', 'legs1_segments0_departureFrom_airport_iata',
    'legs1_segments0_marketingCarrier_code', 'legs1_segments0_operatingCarrier_code',
    'legs1_segments1_aircraft_code', 'legs1_segments1_arrivalTo_airport_city_iata',
    'legs1_segments1_arrivalTo_airport_iata', 'legs1_segments1_departureFrom_airport_iata',
    'legs1_segments1_marketingCarrier_code', 'legs1_segments1_operatingCarrier_code'
]

def duration_to_minutes(duration_str):
    """Convert time format (HH:MM:SS) to minutes"""
    if pd.isna(duration_str) or duration_str is None:
        return np.nan
    try:
        parts = str(duration_str).split(':')
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            return hours * 60 + minutes + seconds / 60
        return np.nan
    except:
        return np.nan

def create_features(df):
    """Create features for flight ranking"""
    # Convert duration columns to minutes
    duration_cols = ['legs0_duration', 'legs1_duration']
    for leg in [0, 1]:
        for seg in range(4):
            duration_cols.append(f'legs{leg}_segments{seg}_duration')
    
    for col in duration_cols:
        if col in df.columns:
            df[col] = df[col].apply(duration_to_minutes)
    
    # Price features
    df['price_per_tax'] = df['totalPrice'] / (df['taxes'] + 1)
    df['tax_rate'] = df['taxes'] / (df['totalPrice'] + 1)
    
    # Duration features
    df['total_duration'] = df['legs0_duration'].fillna(0) + df['legs1_duration'].fillna(0)
    df['duration_ratio'] = df['legs0_duration'] / (df['legs1_duration'].fillna(df['legs0_duration']) + 1)
    
    # Count segments
    for leg in [0, 1]:
        segments = [f'legs{leg}_segments{i}_duration' for i in range(2)]
        df[f'n_segments_leg{leg}'] = df[segments].notna().sum(axis=1)
    df['total_segments'] = df['n_segments_leg0'] + df['n_segments_leg1']
    
    # Trip type
    df['is_one_way'] = df['legs1_duration'].isna().astype(int)
    
    # Ranking features within group
    df['price_rank'] = df.groupby('ranker_id')['totalPrice'].rank()
    df['price_pct_rank'] = df.groupby('ranker_id')['totalPrice'].rank(pct=True)
    df['duration_rank'] = df.groupby('ranker_id')['total_duration'].rank()
    
    # Binary features
    df['frequentFlyer'] = pd.to_numeric(df['frequentFlyer'], errors='coerce').fillna(0)
    df['is_vip_freq'] = ((df['isVip'] == 1) | (df['frequentFlyer'] == 1)).astype(int)
    df['has_return'] = (~df['legs1_duration'].isna()).astype(int)
    df['has_corporate_tariff'] = (~df['corporateTariffCode'].isna()).astype(int)
    
    # Baggage allowance
    df['baggage_total'] = (df['legs0_segments0_baggageAllowance_quantity'].fillna(0) + 
                          df['legs1_segments0_baggageAllowance_quantity'].fillna(0))
    
    # Fees
    df['total_fees'] = (df['miniRules0_monetaryAmount'].fillna(0) + 
                       df['miniRules1_monetaryAmount'].fillna(0))
    df['has_fees'] = (df['total_fees'] > 0).astype(int)
    df['fee_rate'] = df['total_fees'] / (df['totalPrice'] + 1)
    
    # Time features
    for col in ['legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[f'{col}_hour'] = df[col].dt.hour
            df[f'{col}_weekday'] = df[col].dt.weekday
    
    # Direct flight features
    df['is_direct_leg0'] = (df['n_segments_leg0'] == 1).astype(int)
    df['is_direct_leg1'] = (df['n_segments_leg1'] == 1).astype(int)
    df['both_direct'] = (df['is_direct_leg0'] & df['is_direct_leg1']).astype(int)
    
    # Access features
    df['has_access_tp'] = (df['pricingInfo_isAccessTP'] == 1).astype(int)
    
    # Handle categorical NaNs
    for col in cat_features:
        if col in df.columns:
            if df[col].dtype.name == 'Int64':
                df[col] = df[col].astype('Int64').astype(str).replace('<NA>', 'missing')
            else:
                df[col] = df[col].fillna('missing').astype(str)
    
    # Cabin class features
    df['avg_cabin_class'] = df[['legs0_segments0_cabinClass', 'legs1_segments0_cabinClass']].mean(axis=1)
    df['cabin_class_diff'] = df['legs0_segments0_cabinClass'] - df['legs1_segments0_cabinClass']
    
    return df

# Apply feature engineering
train = create_features(train)
test = create_features(test)

# Exclude columns
exclude_cols = ['Id', 'ranker_id', 'selected', 'profileId', 'requestDate',
                'legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt',
                'miniRules0_percentage', 'miniRules1_percentage',  # >90% missing
                'frequentFlyer']  # Already processed

# Exclude segment 2-3 columns (>98% missing)
for leg in [0, 1]:
    for seg in [2, 3]:
        for suffix in ['aircraft_code', 'arrivalTo_airport_city_iata', 'arrivalTo_airport_iata',
                      'baggageAllowance_quantity', 'baggageAllowance_weightMeasurementType',
                      'cabinClass', 'departureFrom_airport_iata', 'duration', 'flightNumber',
                      'marketingCarrier_code', 'operatingCarrier_code', 'seatsAvailable']:
            exclude_cols.append(f'legs{leg}_segments{seg}_{suffix}')

feature_cols = [col for col in train.columns if col not in exclude_cols]
cat_features_final = [col for col in cat_features if col in feature_cols]

print(f"Using {len(feature_cols)} features ({len(cat_features_final)} categorical)")

# Prepare data
X_train = train[feature_cols]
y_train = train['selected']
groups_train = train['ranker_id']

X_test = test[feature_cols]
groups_test = test['ranker_id']

# Group-based split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, val_idx = next(gss.split(X_train, y_train, groups_train))

X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
groups_tr, groups_val = groups_train.iloc[train_idx], groups_train.iloc[val_idx]

print(f"Train: {len(X_tr):,} rows, Val: {len(X_val):,} rows, Test: {len(X_test):,} rows")

# Encode categorical features for XGBoost
for df in [X_tr, X_val, X_test]:
    for col in df.columns:
        if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
            df[col], _ = pd.factorize(df[col])

# Get group sizes for XGBoost
group_sizes_tr = groups_tr.value_counts().sort_index().tolist()
group_sizes_val = groups_val.value_counts().sort_index().tolist()

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_tr, label=y_tr)
dtrain.set_group(group_sizes_tr)

dval = xgb.DMatrix(X_val, label=y_val)
dval.set_group(group_sizes_val)

# XGBoost parameters
params = {
    "objective": "rank:pairwise",
    "learning_rate": 0.1,
    "max_depth": 6,
    "eval_metric": "ndcg@3",
    "random_state": RANDOM_STATE,
    "tree_method": "hist"
}

# Train model
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=200,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=20,
    verbose_eval=10
)

# Generate test predictions
group_sizes_test = groups_test.value_counts().sort_index().tolist()
dtest = xgb.DMatrix(X_test)
dtest.set_group(group_sizes_test)

test_preds = model.predict(dtest)

# Create submission
submission = test[['Id', 'ranker_id']].copy()
submission['pred_score'] = test_preds

# Assign ranks (1 = best option)
submission['selected'] = submission.groupby('ranker_id')['pred_score'].rank(
    ascending=False, method='first'
).astype(int)

# Verify ranking integrity
assert submission.groupby('ranker_id')['selected'].apply(
    lambda x: sorted(x.tolist()) == list(range(1, len(x)+1))
).all(), "Invalid ranking!"

# Save submission
submission[['Id', 'ranker_id', 'selected']].to_csv('submission_xgboost_baseline.csv', index=False)
print(f"Submission saved. Shape: {submission.shape}")