#!/usr/bin/env python3
"""
Generate submission using trained models with exact feature engineering pipeline.
Supports XGBoost, LightGBM, and CatBoost models with command-line model path argument.

Usage:
    python generate_submission.py path/to/model.txt
    python generate_submission.py ./models/lightgbm_ranker_cv_0.58957_20250731_050125.txt
"""

import polars as pl
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import argparse
import sys
import os
from pathlib import Path

def get_model_type_from_path(model_path):
    """Determine model type from file extension"""
    if model_path.endswith('.txt'):
        return 'lightgbm'
    elif model_path.endswith('.json'):
        return 'xgboost'
    elif model_path.endswith('.cbm'):
        return 'catboost'
    else:
        raise ValueError(f"Cannot determine model type from path: {model_path}")

def load_model(model_path, model_type):
    """Load model based on type"""
    if model_type == 'lightgbm':
        return lgb.Booster(model_file=model_path)
    elif model_type == 'xgboost':
        model = xgb.Booster()
        model.load_model(model_path)
        return model
    elif model_type == 'catboost':
        return cb.CatBoost().load_model(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def predict_with_model(model, model_type, X_test, groups_test=None):
    """Make predictions based on model type"""
    if model_type == 'lightgbm':
        return model.predict(X_test.to_numpy(), num_iteration=model.best_iteration)
    elif model_type == 'xgboost':
        # XGBoost needs DMatrix
        group_sizes = groups_test.group_by('ranker_id', maintain_order=True).agg(pl.len())['len'].to_numpy()
        dtest = xgb.DMatrix(X_test, group=group_sizes, feature_names=X_test.columns)
        return model.predict(dtest)
    elif model_type == 'catboost':
        return model.predict(X_test.to_numpy())
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate submission using trained model')
parser.add_argument('model_path', help='Path to the trained model file')
args = parser.parse_args()

# Validate model path
if not os.path.exists(args.model_path):
    print(f"❌ Error: Model file not found: {args.model_path}")
    sys.exit(1)

model_type = get_model_type_from_path(args.model_path)
print(f"🎯 Detected model type: {model_type}")
print(f"📂 Loading model from: {args.model_path}")

print("📊 Loading data and generating submission...")

# Load data
train = pl.read_parquet('./data/train.parquet')
test = pl.read_parquet('./data/test.parquet').with_columns(pl.lit(0, dtype=pl.Int64).alias("selected"))

# Drop __index_level_0__ column if it exists  
if '__index_level_0__' in train.columns:
    train = train.drop('__index_level_0__')
if '__index_level_0__' in test.columns:
    test = test.drop('__index_level_0__')

print(f'Train shape: {train.shape}')
print(f'Test shape: {test.shape}')

data_raw = pl.concat((train, test))

# EXACT FEATURE ENGINEERING FROM train_models.py
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

# EXACT FEATURE SELECTION FROM train_models.py
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

print(f"🔧 Using {len(feature_cols)} features ({len(cat_features_final)} categorical)")

X = data.select(feature_cols)

# Prepare data for the specific model type
if model_type == 'xgboost':
    # XGBoost needs integer encoding
    data_processed = X.with_columns([(pl.col(c).rank("dense") - 1).fill_null(-1).cast(pl.Int16) for c in cat_features_final])
elif model_type == 'lightgbm':
    # LightGBM also needs integer encoding for string categoricals
    data_processed = X.with_columns([(pl.col(c).rank("dense") - 1).fill_null(-1).cast(pl.Int32) for c in cat_features_final])
elif model_type == 'catboost':
    # CatBoost also needs integer encoding for string categoricals
    data_processed = X.with_columns([(pl.col(c).rank("dense") - 1).fill_null(-1).cast(pl.Int32) for c in cat_features_final])
else:
    data_processed = X

# Get test data only
X_test = data_processed[train.height:]
test_groups = test.select(['Id', 'ranker_id'])

print(f"📊 Test data shape: {X_test.shape}")
print(f"📊 Test groups: {len(test_groups.unique('ranker_id'))} unique ranker_ids")

# Load the model
print(f"🤖 Loading {model_type} model...")
model = load_model(args.model_path, model_type)

# Generate predictions
print("🔮 Generating predictions...")
predictions = predict_with_model(model, model_type, X_test, test_groups)

print(f"✅ Generated {len(predictions)} predictions")
print(f"📈 Prediction stats: min={np.min(predictions):.4f}, max={np.max(predictions):.4f}, mean={np.mean(predictions):.4f}")

# Create submission with proper ranking
submission = (
    test_groups.with_columns(pl.Series('pred_score', predictions))
    .with_columns(
        pl.col('pred_score')
        .rank(method='ordinal', descending=True)
        .over('ranker_id')
        .cast(pl.Int32)
        .alias('selected')
    )
    .select(['Id', 'ranker_id', 'selected'])
)

print(f"📋 Submission shape: {submission.shape}")
print("📋 Sample submission:")
print(submission.head(10).to_pandas())

# Verify submission format
unique_rankers = submission.unique('ranker_id').height
total_predictions = submission.height
print(f"📊 Submission validation:")
print(f"   - Unique ranker_ids: {unique_rankers}")
print(f"   - Total predictions: {total_predictions}")
print(f"   - Average options per ranker: {total_predictions/unique_rankers:.1f}")

# Extract model info from filename for submission naming
model_name = Path(args.model_path).stem
submission_file = f'submission_{model_name}.csv'

# Save submission
submission.write_csv(submission_file)
print(f"✅ Submission saved to: {submission_file}")

# Extract CV score from filename if available
cv_score = "unknown"
if "_cv_" in model_name:
    try:
        cv_score = model_name.split("_cv_")[1].split("_")[0]
    except:
        pass

print(f"\n🎯 SUBMISSION READY!")
print(f"📁 File: {submission_file}")
print(f"🤖 Model: {model_type}")
print(f"📊 Local CV: {cv_score}")
print(f"🚀 Ready for Kaggle upload!")