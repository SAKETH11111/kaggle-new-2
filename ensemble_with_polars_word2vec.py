# Ensemble with Polars and Word2Vec Embeddings - FlightRank 2025
# Advanced ensemble approach with Word2Vec embeddings and multiple models

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import time
import xgboost as xgb
import catboost
import lightgbm as lgb
import optuna
from gensim.models import Word2Vec
import gc

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Load data
print("Loading data...")
train = pl.read_parquet('/kaggle/input/aeroclub-recsys-2025/train.parquet').drop('__index_level_0__')
test = pl.read_parquet('/kaggle/input/aeroclub-recsys-2025/test.parquet').drop('__index_level_0__').with_columns(pl.lit(0, dtype=pl.Int64).alias("selected"))

data_raw = pl.concat((train, test))

print("--- Creating Embedding Features with Word2Vec ---")

# Prepare "sentences" for training Word2Vec models
print("Preparing corpus from flight data...")

sentences_df = (
    data_raw
    .group_by("ranker_id")
    .agg([
        pl.col('legs0_segments0_marketingCarrier_code'),
        pl.col('legs0_segments0_departureFrom_airport_iata'),
        pl.col('legs0_segments0_arrivalTo_airport_iata')
    ])
)

corpus = []
for row in sentences_df.iter_rows():
    # Flatten the lists and remove any None values
    sentence = [item for sublist in row[1:] for item in sublist if item is not None]
    if sentence:
        corpus.append(sentence)

print(f"Created {len(corpus)} non-empty sentences for Word2Vec training.")

# Train Word2Vec models
if not corpus:
    raise ValueError("Corpus is empty. Cannot train Word2Vec model.")
    
W2V_PARAMS = {"vector_size": 16, "window": 10, "min_count": 3, "workers": -1, "seed": RANDOM_STATE}

print(f"Training Word2Vec model with params: {W2V_PARAMS}")
w2v_model = Word2Vec(corpus, **W2V_PARAMS)

# Create a dictionary for fast lookups
if not w2v_model.wv.index_to_key:
    print("Warning: Word2Vec model vocabulary is empty. No embeddings will be generated.")
    w2v_dict = {}
else:
    w2v_dict = {word: w2v_model.wv[word] for word in w2v_model.wv.index_to_key}

vector_size = w2v_model.vector_size
default_vector = np.zeros(vector_size, dtype=np.float32)

# Create embedding columns in the main DataFrame
embedding_cols_to_create = {
    'carrier_emb': 'legs0_segments0_marketingCarrier_code',
    'dep_airport_emb': 'legs0_segments0_departureFrom_airport_iata',
    'arr_airport_emb': 'legs0_segments0_arrivalTo_airport_iata'
}

data_with_embeddings = data_raw.clone()

print("Applying embeddings to DataFrame...")
for new_col_prefix, source_col in embedding_cols_to_create.items():
    if source_col in data_with_embeddings.columns:
        # Create a list of vectors first
        embedding_vectors = [w2v_dict.get(key, default_vector) for key in data_with_embeddings[source_col].to_list()]
        
        # Convert list of arrays to a 2D numpy array
        embedding_array = np.array(embedding_vectors)
        
        # Create new columns from the numpy array
        for i in range(vector_size):
            data_with_embeddings = data_with_embeddings.with_columns(
                pl.Series(f"{new_col_prefix}_{i}", embedding_array[:, i])
            )

# Replace the original data variable with the new one
data = data_with_embeddings
print(f"Embedding features created. New DataFrame shape: {data.shape}")

del data_raw, data_with_embeddings, corpus, w2v_model, w2v_dict
gc.collect()

def hitrate_at_3(y_true, y_pred, groups):
    """Calculate HitRate@3 metric"""
    df = pl.DataFrame({
        'group': groups,
        'pred': y_pred,
        'true': y_true
    })
    
    return (
        df.filter(pl.col("group").count().over("group") > 10)
        .sort(["group", "pred"], descending=[False, True])
        .group_by("group", maintain_order=True)
        .head(3)
        .group_by("group")
        .agg(pl.col("true").max())
        .select(pl.col("true").mean())
        .item()
    )

print("--- Feature Engineering ---")

# Make a clone to work on
df = data.clone()

# Duration to minutes converter
def dur_to_min(col):
    days = col.str.extract(r"^(\d+)\.", 1).cast(pl.Int64).fill_null(0) * 1440
    time_str = pl.when(col.str.contains(r"^\d+\.")).then(col.str.replace(r"^\d+\.", "")).otherwise(col)
    hours = time_str.str.extract(r"^(\d+):", 1).cast(pl.Int64).fill_null(0) * 60
    minutes = time_str.str.extract(r":(\d+):", 1).cast(pl.Int64).fill_null(0)
    return (days + hours + minutes).fill_null(0)

# Process duration columns
dur_cols = ["legs0_duration", "legs1_duration"] + [f"legs{l}_segments{s}_duration" for l in (0, 1) for s in (0, 1)]
dur_exprs = [dur_to_min(pl.col(c)).alias(c) for c in dur_cols if c in df.columns]
if dur_exprs:
    df = df.with_columns(dur_exprs)

# Marketing carrier columns check
mc_cols = [f'legs{l}_segments{s}_marketingCarrier_code' for l in (0, 1) for s in range(4)]
mc_exists = [col for col in mc_cols if col in df.columns]

# Initial transformations
df = df.with_columns([
    (pl.col("totalPrice") / (pl.col("taxes") + 1)).alias("price_per_tax"),
    (pl.col("taxes") / (pl.col("totalPrice") + 1)).alias("tax_rate"),
    pl.col("totalPrice").log1p().alias("log_price"),
    (pl.col("legs0_duration").fill_null(0) + pl.col("legs1_duration").fill_null(0)).alias("total_duration"),
    pl.when(pl.col("legs1_duration").fill_null(0) > 0).then(pl.col("legs0_duration") / (pl.col("legs1_duration") + 1)).otherwise(1.0).alias("duration_ratio"),
    (pl.col("legs1_duration").is_null() | (pl.col("legs1_duration") == 0) | pl.col("legs1_segments0_departureFrom_airport_iata").is_null()).cast(pl.Int32).alias("is_one_way"),
    (pl.sum_horizontal(pl.col(col).is_not_null().cast(pl.UInt8) for col in mc_exists) if mc_exists else pl.lit(0)).alias("l0_seg"),
    (pl.col("frequentFlyer").fill_null("").str.count_matches("/") + (pl.col("frequentFlyer").fill_null("") != "").cast(pl.Int32)).alias("n_ff_programs"),
    pl.col("corporateTariffCode").is_not_null().cast(pl.Int32).alias("has_corporate_tariff"),
    (pl.col("pricingInfo_isAccessTP") == 1).cast(pl.Int32).alias("has_access_tp"),
    (pl.col("legs0_segments0_baggageAllowance_quantity").fill_null(0) + pl.col("legs1_segments0_baggageAllowance_quantity").fill_null(0)).alias("baggage_total"),
    (pl.col("miniRules0_monetaryAmount").fill_null(0) + pl.col("miniRules1_monetaryAmount").fill_null(0)).alias("total_fees"),
    pl.col("searchRoute").is_in(["MOWLED/LEDMOW", "LEDMOW/MOWLED", "MOWLED", "LEDMOW", "MOWAER/AERMOW"]).cast(pl.Int32).alias("is_popular_route"),
    pl.mean_horizontal(["legs0_segments0_cabinClass", "legs1_segments0_cabinClass"]).alias("avg_cabin_class"),
    (pl.col("legs0_segments0_cabinClass").fill_null(0) - pl.col("legs1_segments0_cabinClass").fill_null(0)).alias("cabin_class_diff"),
])

# Add more advanced features
df = df.with_columns([
    pl.col("Id").count().over("ranker_id").alias("group_size"),
    pl.col("totalPrice").rank().over("ranker_id").alias("price_rank"),
    pl.col("total_duration").rank().over("ranker_id").alias("duration_rank"),
    (pl.col("totalPrice").rank("average").over("ranker_id") / pl.col("totalPrice").count().over("ranker_id")).alias("price_pct_rank"),
    (pl.col("totalPrice") == pl.col("totalPrice").min().over("ranker_id")).cast(pl.Int32).alias("is_cheapest"),
    ((pl.col("totalPrice") - pl.col("totalPrice").median().over("ranker_id")) / (pl.col("totalPrice").std().over("ranker_id") + 1)).alias("price_from_median"),
])

# Fill nulls
data = df.with_columns(
    [pl.col(c).fill_null(0) for c in df.select(pl.selectors.numeric()).columns] +
    [pl.col(c).fill_null("missing") for c in df.select(pl.selectors.string()).columns]
)

print("--- Feature Selection ---")

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

# Exclude columns
exclude_cols = [
    'Id', 'ranker_id', 'selected', 'profileId', 'requestDate',
    'legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt',
    'miniRules0_percentage', 'miniRules1_percentage',
    'frequentFlyer',
    'pricingInfo_passengerCount'
]

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

print("--- Model Training and Tuning ---")

# Prepare data for XGBoost
data_xgb = X.with_columns([(pl.col(c).rank("dense") - 1).fill_null(-1).cast(pl.Int32) for c in cat_features_final])

n1 = 16487352  # split train to train and val (10%) in time
n2 = train.height
data_xgb_tr, data_xgb_va, data_xgb_te = data_xgb[:n1], data_xgb[n1:n2], data_xgb[n2:]
y_tr, y_va, y_te = y[:n1], y[n1:n2], y[n2:]
groups_tr, groups_va, groups_te = groups[:n1], groups[n1:n2], groups[n2:]

group_sizes_tr = groups_tr.group_by('ranker_id').agg(pl.len()).sort('ranker_id')['len'].to_numpy()
group_sizes_va = groups_va.group_by('ranker_id').agg(pl.len()).sort('ranker_id')['len'].to_numpy()
group_sizes_te = groups_te.group_by('ranker_id').agg(pl.len()).sort('ranker_id')['len'].to_numpy()

dtrain = xgb.DMatrix(data_xgb_tr, label=y_tr, group=group_sizes_tr, feature_names=data_xgb.columns)
dval = xgb.DMatrix(data_xgb_va, label=y_va, group=group_sizes_va, feature_names=data_xgb.columns)
dtest = xgb.DMatrix(data_xgb_te, label=y_te, group=group_sizes_te, feature_names=data_xgb.columns)

# XGBoost parameters (optimized)
final_xgb_params = {
    'objective': 'rank:pairwise', 
    'eval_metric': 'ndcg@3', 
    'max_depth': 8, 
    'min_child_weight': 14, 
    'subsample': 0.9, 
    'colsample_bytree': 1.0, 
    'lambda': 3.5330891736457763, 
    'learning_rate': 0.0521879929228514,
    'seed': RANDOM_STATE, 
    'n_jobs': -1
}

print("Training XGBoost model with optimized parameters...")
xgb_model = xgb.train(
    final_xgb_params, dtrain,
    num_boost_round=1500,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=100,
    verbose_eval=50
)

# LightGBM Model
print("Creating LightGBM Datasets...")
X_tr, X_va, X_te = X[:n1], X[n1:n2], X[n2:]

lgb_train = lgb.Dataset(
    data=X_tr.to_pandas(), 
    label=y_tr.to_pandas(), 
    group=group_sizes_tr,
    feature_name=X.columns,
    categorical_feature=cat_features_final,
    free_raw_data=False
)

lgb_val = lgb.Dataset(
    data=X_va.to_pandas(), 
    label=y_va.to_pandas(), 
    group=group_sizes_va,
    feature_name=X.columns,
    categorical_feature=cat_features_final, 
    reference=lgb_train,
    free_raw_data=False
)

# LightGBM parameters (optimized)
final_lgb_params = {
    'objective': 'lambdarank', 
    'metric': 'ndcg', 
    'boosting_type': 'gbdt',
    'eval_at': [3],
    'num_leaves': 137, 
    'learning_rate': 0.19236092380700556, 
    'min_child_samples': 69, 
    'lambda_l1': 0.001786334561662628, 
    'lambda_l2': 7.881799636447006, 
    'feature_fraction': 0.6015465928218717, 
    'bagging_fraction': 0.8535794374747682, 
    'bagging_freq': 7, 
    'n_jobs': -1, 
    'random_state': RANDOM_STATE, 
    'label_gain': [0, 1]
}

print("Training LightGBM model with optimized parameters...")
lgb_model = lgb.train(
    final_lgb_params,
    lgb_train,
    num_boost_round=1500,
    valid_sets=[lgb_train, lgb_val],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)]
)

# DART Model
print("Training LightGBM DART Model...")
dart_params = {
    'objective': 'lambdarank', 
    'metric': 'ndcg', 
    'eval_at': [3],
    'boosting_type': 'dart', 
    'n_estimators': 1500,     
    'learning_rate': 0.04,
    'num_leaves': 50,
    'drop_rate': 0.1,        
    'skip_drop': 0.5,        
    'n_jobs': -1,
    'random_state': RANDOM_STATE,
    'label_gain': [0, 1]
}

lgb_model_dart = lgb.train(
    dart_params,
    lgb_train, 
    num_boost_round=dart_params['n_estimators'], 
    valid_sets=[lgb_val],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)],
    categorical_feature=cat_features_final 
)

print("--- Blending and Final Evaluation ---")

# Get predictions from all models on validation set
X_va_pd = X_va.to_pandas()
X_va_num_pd = data_xgb_va.to_pandas()  # For XGBoost

xgb_va_preds = xgb_model.predict(xgb.DMatrix(X_va_num_pd))
lgb_va_preds = lgb_model.predict(X_va_pd)
lgb_dart_va_preds = lgb_model_dart.predict(X_va_pd)

# Calculate smart blending weights based on individual validation performance
xgb_hr3 = hitrate_at_3(y_va['selected'], xgb_va_preds, groups_va['ranker_id'])
lgb_hr3 = hitrate_at_3(y_va['selected'], lgb_va_preds, groups_va['ranker_id'])
lgb_dart_hr3 = hitrate_at_3(y_va['selected'], lgb_dart_va_preds, groups_va['ranker_id'])

total_hr3 = xgb_hr3 + lgb_hr3 + lgb_dart_hr3 + 1e-9 
w_xgb = xgb_hr3 / total_hr3
w_lgb = lgb_hr3 / total_hr3
w_dart = lgb_dart_hr3 / total_hr3

print(f"XGBoost HitRate@3:     {xgb_hr3:.5f}")
print(f"LGBM GBDT HitRate@3:   {lgb_hr3:.5f}")
print(f"LGBM DART HitRate@3:   {lgb_dart_hr3:.5f}")
print(f"Blending Weights (XGB/LGB/DART): {w_xgb:.3f} / {w_lgb:.3f} / {w_dart:.3f}")

print("--- Generating Final Submission ---")

# Generate test predictions
X_te_pd = X_te.to_pandas()
X_te_num_pd = data_xgb_te.to_pandas()

xgb_test_preds = xgb_model.predict(xgb.DMatrix(X_te_num_pd))
lgb_gbdt_test_preds = lgb_model.predict(X_te_pd) 
lgb_dart_test_preds = lgb_model_dart.predict(X_te_pd)

# Create submission with smart blending
submission_df = test.select(['Id', 'ranker_id']).with_columns(
    pl.Series('xgb_score', xgb_test_preds),
    pl.Series('lgb_gbdt_score', lgb_gbdt_test_preds),
    pl.Series('lgb_dart_score', lgb_dart_test_preds)
).with_columns(
    xgb_rank=pl.col("xgb_score").rank(method="average", descending=True).over("ranker_id"),
    lgb_gbdt_rank=pl.col("lgb_gbdt_score").rank(method="average", descending=True).over("ranker_id"),
    lgb_dart_rank=pl.col("lgb_dart_score").rank(method="average", descending=True).over("ranker_id")
).with_columns(
    blend_score=(w_xgb * pl.col("xgb_rank") + w_lgb * pl.col("lgb_gbdt_rank") + w_dart * pl.col("lgb_dart_rank"))
).with_columns(
    selected=pl.col('blend_score').rank(method='ordinal', descending=False).over('ranker_id').cast(pl.Int32)
).select(['Id', 'ranker_id', 'selected'])

submission_df.write_csv('submission_ensemble_polars_word2vec.csv')

print("Submission file 'submission_ensemble_polars_word2vec.csv' created successfully.")
print(f"Submission shape: {submission_df.shape}")