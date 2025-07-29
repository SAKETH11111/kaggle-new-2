# -*- coding: utf-8 -*-
"""FlightRank 2025 - Prediction Script
Loads the trained model and generates predictions for test data.
"""

import polars as pl
import numpy as np
import xgboost as xgb
import pickle
import os

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def load_and_engineer_features(data_path):
    """Apply the exact same feature engineering as in train_and_validate.py"""
    
    print(f'Loading data from {data_path}...')
    
    # Load test data
    test = pl.read_parquet(data_path).with_columns(pl.lit(0, dtype=pl.Int64).alias("selected"))
    
    # Drop __index_level_0__ column if it exists
    if '__index_level_0__' in test.columns:
        test = test.drop('__index_level_0__')
    
    print(f'Test shape: {test.shape}')
    
    # Start feature engineering (identical to train_and_validate.py)
    df = test.clone()
    
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
        
        # Fees features
        (
            (pl.col("miniRules0_monetaryAmount") == 0)
            & (pl.col("miniRules0_statusInfos") == 1)
        ).cast(pl.Int8).alias("free_cancel"),
        (
            (pl.col("miniRules1_monetaryAmount") == 0)
            & (pl.col("miniRules1_statusInfos") == 1)
        ).cast(pl.Int8).alias("free_exchange"),
        
        # Routes & carriers
        pl.col("searchRoute").is_in(["MOWLED/LEDMOW", "LEDMOW/MOWLED", "MOWLED", "LEDMOW"])
            .cast(pl.Int32).alias("is_popular_route"),
        
        # Cabin
        pl.mean_horizontal(["legs0_segments0_cabinClass", "legs1_segments0_cabinClass"]).alias("avg_cabin_class"),
        (pl.col("legs0_segments0_cabinClass").fill_null(0) -
         pl.col("legs1_segments0_cabinClass").fill_null(0)).alias("cabin_class_diff"),
    ])
    
    # Segment counts
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
    
    df = df.with_columns(seg_exprs)
    
    # Derived features
    df = df.with_columns([
        (pl.col("n_segments_leg0") + pl.col("n_segments_leg1")).alias("total_segments"),
        (pl.col("n_segments_leg0") == 1).cast(pl.Int32).alias("is_direct_leg0"),
        pl.when(pl.col("is_one_way") == 1).then(0)
            .otherwise((pl.col("n_segments_leg1") == 1).cast(pl.Int32)).alias("is_direct_leg1"),
    ])
    
    # More features
    df = df.with_columns([
        (pl.col("is_direct_leg0") & pl.col("is_direct_leg1")).cast(pl.Int32).alias("both_direct"),
        ((pl.col("isVip") == 1) | (pl.col("n_ff_programs") > 0)).cast(pl.Int32).alias("is_vip_freq"),
        pl.col("Id").count().over("ranker_id").alias("group_size"),
    ])
    
    # Major carrier flag
    if "legs0_segments0_marketingCarrier_code" in df.columns:
        df = df.with_columns(
            pl.col("legs0_segments0_marketingCarrier_code").is_in(["SU", "S7"])
                .cast(pl.Int32).alias("is_major_carrier")
        )
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
    
    # Rank computations
    df = df.with_columns([
        pl.col("group_size").log1p().alias("group_size_log"),
    ])
    
    # Price and duration ranks
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
    
    df = df.with_columns(rank_exprs + price_exprs)
    
    # Cheapest direct
    direct_cheapest = (
        df.filter(pl.col("is_direct_leg0") == 1)
        .group_by("ranker_id")
        .agg(pl.col("totalPrice").min().alias("min_direct"))
    )
    
    df = df.join(direct_cheapest, on="ranker_id", how="left").with_columns(
        ((pl.col("is_direct_leg0") == 1) &
         (pl.col("totalPrice") == pl.col("min_direct"))).cast(pl.Int32).fill_null(0).alias("is_direct_cheapest")
    ).drop("min_direct")
    
    # For test data, we need to create dummy popularity features (can't use train data here)
    df = df.with_columns([
        pl.lit(0.0).alias('carrier0_pop'),
        pl.lit(0.0).alias('carrier1_pop'),
    ])
    
    # Final popularity features
    df = df.with_columns([
        (pl.col('carrier0_pop') * pl.col('carrier1_pop')).alias('carrier_pop_product'),
    ])
    
    # Gap features (the critical ones that improved performance)
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
        ((pl.col("is_direct_leg0") == 1) & 
         (pl.col("totalPrice") == pl.col("min_direct_price")) &
         pl.col("min_direct_price").is_not_null())
        .cast(pl.Int32)
        .fill_null(0)
        .alias("is_best_direct")
    ]).drop("min_direct_price")
    
    # Fill nulls
    data = df.with_columns(
        [pl.col(c).fill_null(0) for c in df.select(pl.selectors.numeric()).columns] +
        [pl.col(c).fill_null("missing") for c in df.select(pl.selectors.string()).columns]
    )
    
    return data

def main():
    """Main prediction pipeline"""
    
    print("🚀 FlightRank 2025 - Prediction Script")
    print("=" * 50)
    
    # Check if model exists
    model_path = './models/best_model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please run train_and_validate.py first.")
    
    # Load the trained model
    print("📦 Loading trained model...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully")
    
    # Load and engineer test features
    test_data = load_and_engineer_features('./data/test.parquet')
    
    # Feature selection (same as training)
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
    
    # Add segment 2-3 columns to exclude
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
    
    for leg in [0, 1]:
        for seg in [2, 3]:
            for suffix in ['aircraft_code', 'arrivalTo_airport_city_iata', 'arrivalTo_airport_iata',
                          'baggageAllowance_quantity', 'baggageAllowance_weightMeasurementType',
                          'cabinClass', 'departureFrom_airport_iata', 'duration', 'flightNumber',
                          'marketingCarrier_code', 'operatingCarrier_code', 'seatsAvailable']:
                exclude_cols.append(f'legs{leg}_segments{seg}_{suffix}')
    
    feature_cols = [col for col in test_data.columns if col not in exclude_cols]
    cat_features_final = [col for col in cat_features if col in feature_cols]
    
    print(f"Using {len(feature_cols)} features ({len(cat_features_final)} categorical)")
    
    # Prepare features for prediction
    X_test = test_data.select(feature_cols)
    groups_test = test_data.select('ranker_id')
    
    # Encode categorical features
    data_xgb_test = X_test.with_columns([(pl.col(c).rank("dense") - 1).fill_null(-1).cast(pl.Int16) for c in cat_features_final])
    
    # Prepare for XGBoost (sort by groups)
    groups_test_np = groups_test.to_numpy().flatten()
    unique_groups_test, group_sizes_test = np.unique(groups_test_np, return_counts=True)
    test_sort_idx = np.argsort(groups_test_np)
    X_test_sorted = data_xgb_test.to_numpy()[test_sort_idx]
    
    # Create DMatrix and predict
    print("🔮 Generating predictions...")
    dtest = xgb.DMatrix(X_test_sorted, group=group_sizes_test, feature_names=data_xgb_test.columns)
    test_predictions = model.predict(dtest)
    
    # Restore original order
    reverse_sort_idx = np.argsort(test_sort_idx)
    test_predictions_original_order = test_predictions[reverse_sort_idx]
    
    # Create submission with rule-based re-ranking
    def re_rank(test_df, pred_scores, penalty_factor=0.1):
        """Apply rule-based re-ranking (same as training)"""
        COLS_TO_COMPARE = [
            "legs0_departureAt", "legs0_arrivalAt", "legs1_departureAt", "legs1_arrivalAt",
            "legs0_segments0_flightNumber", "legs1_segments0_flightNumber",
            "legs0_segments0_aircraft_code", "legs1_segments0_aircraft_code",
            "legs0_segments0_departureFrom_airport_iata", "legs1_segments0_departureFrom_airport_iata",
        ]
        
        submission_df = test_df.select(['Id', 'ranker_id']).with_columns(
            pl.Series('pred_score', pred_scores)
        )
        
        test_with_scores = submission_df.join(test_df, on=["Id", "ranker_id"], how="left")
        
        test_with_scores = test_with_scores.with_columns(
            [pl.col(c).cast(str).fill_null("NULL") for c in COLS_TO_COMPARE if c in test_with_scores.columns]
        )
        
        if all(c in test_with_scores.columns for c in COLS_TO_COMPARE):
            test_with_scores = test_with_scores.with_columns(
                (
                    pl.col("legs0_departureAt") + "_" + pl.col("legs0_arrivalAt") + "_" +
                    pl.col("legs1_departureAt") + "_" + pl.col("legs1_arrivalAt") + "_" +
                    pl.col("legs0_segments0_flightNumber") + "_" + pl.col("legs1_segments0_flightNumber")
                ).alias("flight_hash")
            )
            
            test_with_scores = test_with_scores.with_columns(
                pl.max("pred_score").over(["ranker_id", "flight_hash"]).alias("max_score_same_flight")
            )
            
            test_with_scores = test_with_scores.with_columns(
                (pl.col("pred_score") - penalty_factor * (pl.col("max_score_same_flight") - pl.col("pred_score"))).alias("reorder_score")
            )
            
            final_submission = test_with_scores.with_columns(
                pl.col("reorder_score").rank(method="ordinal", descending=True).over("ranker_id").cast(pl.Int32).alias("selected")
            ).select(["Id", "ranker_id", "selected"])
        else:
            # Fallback if re-ranking columns are missing
            final_submission = submission_df.with_columns(
                pl.col("pred_score").rank(method="ordinal", descending=True).over("ranker_id").cast(pl.Int32).alias("selected")
            ).select(["Id", "ranker_id", "selected"])
        
        return final_submission
    
    # Generate final submission
    print("📊 Applying rule-based re-ranking...")
    submission = re_rank(test_data, test_predictions_original_order)
    
    # Save submission
    submission.write_csv('submission.csv')
    print("✅ Submission saved to submission.csv")
    print(f"📋 Submission shape: {submission.shape}")
    print("🎯 Ready for competition submission!")

if __name__ == "__main__":
    main()