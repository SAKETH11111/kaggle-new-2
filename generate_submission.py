#!/usr/bin/env python3
"""Generate submission using the trained model with corrected CV framework"""

import polars as pl
import xgboost as xgb
import numpy as np

print("Loading test data and generating submission...")

# Load test data
test = pl.read_parquet('./data/test.parquet').with_columns(pl.lit(0, dtype=pl.Int64).alias("selected"))
train = pl.read_parquet('./data/train.parquet')  # Needed for feature engineering

# Drop __index_level_0__ column if it exists
if '__index_level_0__' in test.columns:
    test = test.drop('__index_level_0__')
if '__index_level_0__' in train.columns:
    train = train.drop('__index_level_0__')

print(f'Test shape: {test.shape}')

# Recreate the same feature engineering pipeline from baseline_with_cv.py
data_raw = pl.concat((train, test))
df = data_raw.clone()

# Feature engineering (same as in baseline_with_cv.py)
def dur_to_min(col):
    days = col.str.extract(r"^(\d+)\.", 1).cast(pl.Int64).fill_null(0) * 1440
    time_str = pl.when(col.str.contains(r"^\d+\.")).then(col.str.replace(r"^\d+\.", "")).otherwise(col)
    hours = time_str.str.extract(r"^(\d+):", 1).cast(pl.Int64).fill_null(0) * 60
    minutes = time_str.str.extract(r":(\d+):", 1).cast(pl.Int64).fill_null(0)
    return (days + hours + minutes).fill_null(0)

dur_cols = ["legs0_duration", "legs1_duration"] + [f"legs{l}_segments{s}_duration" for l in (0, 1) for s in (0, 1)]
dur_exprs = [dur_to_min(pl.col(c)).alias(c) for c in dur_cols if c in df.columns]

if dur_exprs:
    df = df.with_columns(dur_exprs)

mc_cols = [f'legs{l}_segments{s}_marketingCarrier_code' for l in (0, 1) for s in range(4)]
mc_exists = [col for col in mc_cols if col in df.columns]

# Apply all transformations (same as baseline_with_cv.py)
df = df.with_columns([
    (pl.col("taxes") / (pl.col("totalPrice") + 1)).alias("tax_rate"),
    pl.col("totalPrice").log1p().alias("log_price"),
    (pl.col("legs0_duration").fill_null(0) + pl.col("legs1_duration").fill_null(0)).alias("total_duration"),
    pl.when(pl.col("legs1_duration").fill_null(0) > 0)
        .then(pl.col("legs0_duration") / (pl.col("legs1_duration") + 1))
        .otherwise(1.0).alias("duration_ratio"),
    (pl.col("legs1_duration").is_null() |
     (pl.col("legs1_duration") == 0) |
     pl.col("legs1_segments0_departureFrom_airport_iata").is_null()).cast(pl.Int32).alias("is_one_way"),
    (pl.sum_horizontal(pl.col(col).is_not_null().cast(pl.UInt8) for col in mc_exists)
     if mc_exists else pl.lit(0)).alias("l0_seg"),
    (pl.col("frequentFlyer").fill_null("").str.count_matches("/") +
     (pl.col("frequentFlyer").fill_null("") != "").cast(pl.Int32)).alias("n_ff_programs"),
    pl.col("corporateTariffCode").is_not_null().cast(pl.Int32).alias("has_corporate_tariff"),
    (pl.col("pricingInfo_isAccessTP") == 1).cast(pl.Int32).alias("has_access_tp"),
    ((pl.col("miniRules0_monetaryAmount") == 0) & (pl.col("miniRules0_statusInfos") == 1)).cast(pl.Int8).alias("free_cancel"),
    ((pl.col("miniRules1_monetaryAmount") == 0) & (pl.col("miniRules1_statusInfos") == 1)).cast(pl.Int8).alias("free_exchange"),
    pl.col("searchRoute").is_in(["MOWLED/LEDMOW", "LEDMOW/MOWLED", "MOWLED", "LEDMOW"]).cast(pl.Int32).alias("is_popular_route"),
    pl.mean_horizontal(["legs0_segments0_cabinClass", "legs1_segments0_cabinClass"]).alias("avg_cabin_class"),
    (pl.col("legs0_segments0_cabinClass").fill_null(0) - pl.col("legs1_segments0_cabinClass").fill_null(0)).alias("cabin_class_diff"),
])

# Segment features
seg_exprs = []
for leg in (0, 1):
    seg_cols = [f"legs{leg}_segments{s}_duration" for s in range(4) if f"legs{leg}_segments{s}_duration" in df.columns]
    if seg_cols:
        seg_exprs.append(pl.sum_horizontal(pl.col(c).is_not_null() for c in seg_cols).cast(pl.Int32).alias(f"n_segments_leg{leg}"))
    else:
        seg_exprs.append(pl.lit(0).cast(pl.Int32).alias(f"n_segments_leg{leg}"))

df = df.with_columns(seg_exprs)

df = df.with_columns([
    (pl.col("n_segments_leg0") + pl.col("n_segments_leg1")).alias("total_segments"),
    (pl.col("n_segments_leg0") == 1).cast(pl.Int32).alias("is_direct_leg0"),
    pl.when(pl.col("is_one_way") == 1).then(0).otherwise((pl.col("n_segments_leg1") == 1).cast(pl.Int32)).alias("is_direct_leg1"),
])

df = df.with_columns([
    (pl.col("is_direct_leg0") & pl.col("is_direct_leg1")).cast(pl.Int32).alias("both_direct"),
    ((pl.col("isVip") == 1) | (pl.col("n_ff_programs") > 0)).cast(pl.Int32).alias("is_vip_freq"),
    pl.col("Id").count().over("ranker_id").alias("group_size"),
])

if "legs0_segments0_marketingCarrier_code" in df.columns:
    df = df.with_columns(pl.col("legs0_segments0_marketingCarrier_code").is_in(["SU", "S7"]).cast(pl.Int32).alias("is_major_carrier"))
else:
    df = df.with_columns(pl.lit(0).alias("is_major_carrier"))

df = df.with_columns(pl.col("group_size").log1p().alias("group_size_log"))

# Time features
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

# Ranking features
rank_exprs = []
for col, alias in [("totalPrice", "price"), ("total_duration", "duration")]:
    rank_exprs.append(pl.col(col).rank().over("ranker_id").alias(f"{alias}_rank"))

price_exprs = [
    (pl.col("totalPrice").rank("average").over("ranker_id") / pl.col("totalPrice").count().over("ranker_id")).alias("price_pct_rank"),
    (pl.col("totalPrice") == pl.col("totalPrice").min().over("ranker_id")).cast(pl.Int32).alias("is_cheapest"),
    ((pl.col("totalPrice") - pl.col("totalPrice").median().over("ranker_id")) / (pl.col("totalPrice").std().over("ranker_id") + 1)).alias("price_from_median"),
    (pl.col("l0_seg") == pl.col("l0_seg").min().over("ranker_id")).cast(pl.Int32).alias("is_min_segments"),
]

df = df.with_columns(rank_exprs + price_exprs)

# Direct cheapest
direct_cheapest = df.filter(pl.col("is_direct_leg0") == 1).group_by("ranker_id").agg(pl.col("totalPrice").min().alias("min_direct"))
df = df.join(direct_cheapest, on="ranker_id", how="left").with_columns(
    ((pl.col("is_direct_leg0") == 1) & (pl.col("totalPrice") == pl.col("min_direct"))).cast(pl.Int32).fill_null(0).alias("is_direct_cheapest")
).drop("min_direct")

# Popularity features
df = (df.join(train.group_by('legs0_segments0_marketingCarrier_code').agg(pl.mean('selected').alias('carrier0_pop')), on='legs0_segments0_marketingCarrier_code', how='left')
      .join(train.group_by('legs1_segments0_marketingCarrier_code').agg(pl.mean('selected').alias('carrier1_pop')), on='legs1_segments0_marketingCarrier_code', how='left')
      .with_columns([pl.col('carrier0_pop').fill_null(0.0), pl.col('carrier1_pop').fill_null(0.0)]))

df = df.with_columns([(pl.col('carrier0_pop') * pl.col('carrier1_pop')).alias('carrier_pop_product')])

# Gap features
df = df.with_columns([
    (pl.col("totalPrice") - pl.col("totalPrice").min().over("ranker_id")).alias("price_gap"),
    (pl.col("legs0_departureAt_hour") - pl.col("legs0_departureAt_hour").min().over("ranker_id")).alias("departure_gap").fill_null(0),
    pl.when(pl.col("is_direct_leg0") == 1).then(pl.col("totalPrice")).otherwise(None).min().over("ranker_id").alias("min_direct_price"),
]).with_columns([
    ((pl.col("is_direct_leg0") == 1) & (pl.col("totalPrice") == pl.col("min_direct_price")) & pl.col("min_direct_price").is_not_null()).cast(pl.Int32).fill_null(0).alias("is_best_direct")
]).drop("min_direct_price")

# Fill nulls
data = df.with_columns(
    [pl.col(c).fill_null(0) for c in df.select(pl.selectors.numeric()).columns] +
    [pl.col(c).fill_null("missing") for c in df.select(pl.selectors.string()).columns]
)

# Feature selection (same as baseline_with_cv.py)
cat_features = [
    'nationality', 'searchRoute', 'corporateTariffCode', 'bySelf', 'sex', 'companyID',
    'legs0_segments0_aircraft_code', 'legs0_segments0_arrivalTo_airport_city_iata',
    'legs0_segments0_arrivalTo_airport_iata', 'legs0_segments0_departureFrom_airport_iata',
    'legs0_segments0_marketingCarrier_code', 'legs0_segments0_operatingCarrier_code',
    'legs0_segments0_flightNumber', 'legs0_segments1_aircraft_code', 'legs0_segments1_arrivalTo_airport_city_iata',
    'legs0_segments1_arrivalTo_airport_iata', 'legs0_segments1_departureFrom_airport_iata',
    'legs0_segments1_marketingCarrier_code', 'legs0_segments1_operatingCarrier_code',
    'legs0_segments1_flightNumber', 'legs1_segments0_aircraft_code', 'legs1_segments0_arrivalTo_airport_city_iata',
    'legs1_segments0_arrivalTo_airport_iata', 'legs1_segments0_departureFrom_airport_iata',
    'legs1_segments0_marketingCarrier_code', 'legs1_segments0_operatingCarrier_code',
    'legs1_segments0_flightNumber', 'legs1_segments1_aircraft_code', 'legs1_segments1_arrivalTo_airport_city_iata',
    'legs1_segments1_arrivalTo_airport_iata', 'legs1_segments1_departureFrom_airport_iata',
    'legs1_segments1_marketingCarrier_code', 'legs1_segments1_operatingCarrier_code', 'legs1_segments1_flightNumber',
]

exclude_cols = [
    'Id', 'ranker_id', 'selected', 'profileId', 'requestDate',
    'legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt',
    'miniRules0_percentage', 'miniRules1_percentage', 'frequentFlyer', 'pricingInfo_passengerCount'
]

for leg in [0, 1]:
    for seg in [0, 1]:
        if seg == 0:
            suffixes = ["seatsAvailable"]
        else:
            suffixes = ["cabinClass", "seatsAvailable", "baggageAllowance_quantity", "baggageAllowance_weightMeasurementType", 
                       "aircraft_code", "arrivalTo_airport_city_iata", "arrivalTo_airport_iata", "departureFrom_airport_iata",
                       "flightNumber", "marketingCarrier_code", "operatingCarrier_code"]
        for suffix in suffixes:
            exclude_cols.append(f"legs{leg}_segments{seg}_{suffix}")

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

# Prepare test data
X = data.select(feature_cols)
data_xgb = X.with_columns([(pl.col(c).rank("dense") - 1).fill_null(-1).cast(pl.Int16) for c in cat_features_final])

# Get test data only
X_test = data_xgb[train.height:]
test_data = test.select(['Id', 'ranker_id'])

print(f"Test data shape: {X_test.shape}")

# Load the trained model (full data version)
model_path = './models/xgboost_ranker_cv_0.58139_20250730_054315.json'
print(f"Loading model from: {model_path}")

model = xgb.Booster()
model.load_model(model_path)

# Create DMatrix for prediction
group_sizes_test = test_data.group_by('ranker_id', maintain_order=True).agg(pl.len())['len'].to_numpy()
dtest = xgb.DMatrix(X_test, group=group_sizes_test, feature_names=X_test.columns)

print("Generating predictions...")
predictions = model.predict(dtest)

# Create submission
submission = (
    test_data.with_columns(pl.Series('pred_score', predictions))
    .with_columns(
        pl.col('pred_score')
        .rank(method='ordinal', descending=True)
        .over('ranker_id')
        .cast(pl.Int32)
        .alias('selected')
    )
    .select(['Id', 'ranker_id', 'selected'])
)

print(f"Submission shape: {submission.shape}")
print("Sample submission:")
print(submission.head(10).to_pandas())

# Save submission
submission.write_csv('submission_cv_0581_fulldata.csv')
print("✅ Submission saved to: submission_cv_0581_fulldata.csv")

print(f"\n🎯 Ready to submit! CV: 0.581 (100% data)")
print(f"Expected public score: Unknown - CV still seems inflated")
print(f"This should perform better than our 0.427 score with 20% data")