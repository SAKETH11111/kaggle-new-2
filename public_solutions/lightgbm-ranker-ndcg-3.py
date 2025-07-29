# %%
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import gc # Garbage Collector
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# %%
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print(f'Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

def calculate_hit_rate_at_3(df_preds_with_true_and_rank):
    """
    Calculates HitRate@3.
    df_preds_with_true_and_rank must have:
        - 'ranker_id'
        - 'selected' (true binary target, 1 for chosen)
        - 'predicted_rank' (rank assigned by the model, 1 is best)
    """
    hits = 0
    valid_queries_count = 0
    
    for ranker_id, group in df_preds_with_true_and_rank.groupby('ranker_id'):
        if len(group) <= 10:
            continue  # Skip groups with 10 or fewer options as per competition rules
        
        valid_queries_count += 1
        
        true_selected_item = group[group['selected'] == 1]
        
        if not true_selected_item.empty:
            # Get the rank of the true selected item
            rank_of_true_item = true_selected_item.iloc[0]['predicted_rank']
            if rank_of_true_item <= 3:
                hits += 1
        # else:
            # This shouldn't happen in validation if data is prepared correctly from train
            # print(f"Warning: No selected item found for ranker_id {ranker_id} in HitRate calculation.")
            
    if valid_queries_count == 0:
        return 0.0
    return hits / valid_queries_count

# %%
# Cell 3: Load Data
import pandas as pd
import numpy as np
import gc

# DEFINE CORE COLUMNS TO LOAD INITIALLY (TỐI GIẢN HÓA + BỔ SUNG)
initial_core_columns = [
    'Id', 'ranker_id', 'selected', 'profileId', 'companyID',
    'requestDate', 'totalPrice', 'taxes', # Cần cho totalPrice_rank_in_group và price_per_tax, tax_ratio
    'legs0_departureAt', 'legs0_arrivalAt', 'legs0_duration',
    'legs1_departureAt', 'legs1_arrivalAt', 'legs1_duration', 
    # Removed legs1_segments0_departureFrom_airport_iata, legs0_segments_X... as num_segments is derived differently
    'legs0_segments0_departureFrom_airport_iata', # Base for segment count logic
    'searchRoute', # Cho is_round_trip
    'pricingInfo_isAccessTP', # Cho is_compliant
    'sex', 'nationality', 'isVip', # User info cơ bản
    
    # Bổ sung từ CatBoost ideas
    'legs0_segments0_cabinClass', 'legs1_segments0_cabinClass', # Cabin class features
    'miniRules0_monetaryAmount', 'miniRules1_monetaryAmount', # Fee features
    'miniRules0_percentage', 'miniRules1_percentage', # Original free_cancel/exchange features
    'legs0_segments0_baggageAllowance_quantity', # Baggage total
    'legs1_segments0_baggageAllowance_quantity',
    'corporateTariffCode', # has_corporate_tariff
    'frequentFlyer', # frequentFlyer_binary
]
# Add all segment related columns up to segment 3 for more robust num_segments_legX calculation
for leg_idx in [0, 1]:
    for seg_idx in range(4): # Catboost went up to 3, let's keep 4 for now
        initial_core_columns.append(f'legs{leg_idx}_segments{seg_idx}_departureFrom_airport_iata')

initial_core_columns = list(set(initial_core_columns)) # Remove duplicates if any

initial_core_columns_test = [col for col in initial_core_columns if col != 'selected']


print("Loading a subset of columns for train_df...")
train_df = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/train.parquet', columns=initial_core_columns)
print("Loading a subset of columns for test_df...")
test_df = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/test.parquet', columns=initial_core_columns_test)
sample_submission_df = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/sample_submission.parquet')

print("\nTrain DataFrame (after loading subset - BEFORE reduce_mem_usage and any FE):")
train_df.info(memory_usage='deep')
print(f"\nShape: {train_df.shape}")
print("\nTest DataFrame (after loading subset - BEFORE reduce_mem_usage and any FE):")
test_df.info(memory_usage='deep')
print(f"\nShape: {test_df.shape}")

if 'Id' in test_df.columns and 'ranker_id' in test_df.columns:
    test_ids_df = test_df[['Id', 'ranker_id']].copy()
else:
    print("Warning: 'Id' or 'ranker_id' not found in loaded test_df columns.")
    test_ids_df = pd.DataFrame()
gc.collect()

# %%
# Cell 4: Feature Engineering 

def create_initial_datetime_features(df):
    loaded_cols = df.columns
    potential_dt_cols = ['requestDate', 'legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt']
    for col in potential_dt_cols:
        if col in loaded_cols:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col].astype(str), errors='coerce')
    return df

def create_remaining_features(df, is_train=True):
    print(f"Starting FE. Initial df memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # --- Basic time, booking, route, segment features ---
    potential_dt_cols_for_components = ['legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt']
    for col in potential_dt_cols_for_components:
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
             # Extract components, fill NaNs resulting from NaT, then cast
             df[col + '_hour'] = df[col].dt.hour.fillna(-1).astype(np.int8) # Fill NaN with -1
             df[col + '_dow'] = df[col].dt.dayofweek.fillna(-1).astype(np.int8) # Fill NaN with -1
    
    if 'legs0_departureAt' in df.columns and 'requestDate' in df.columns and \
       pd.api.types.is_datetime64_any_dtype(df['legs0_departureAt']) and \
       pd.api.types.is_datetime64_any_dtype(df['requestDate']):
        # Calculate booking_lead_days only if both columns are valid datetimes
        # NaT in legs0_departureAt or requestDate will result in NaT for the difference,
        # then .dt.total_seconds() will produce NaN.
        time_diff = (df['legs0_departureAt'] - df['requestDate'])
        df['booking_lead_days'] = time_diff.dt.total_seconds() / (24 * 60 * 60)
        df['booking_lead_days'] = df['booking_lead_days'].fillna(-1).astype(np.float32)
    else: 
        df['booking_lead_days'] = np.float32(-1.0)
    
    if 'searchRoute' in df.columns: 
        df['is_round_trip'] = df['searchRoute'].astype(str).str.contains('/').astype(np.int8)
    else: 
        df['is_round_trip'] = np.int8(-1)
    
    if 'legs1_departureAt' in df.columns:
        # Ensure it's datetime before checking .notna() for num_legs calculation
        if not pd.api.types.is_datetime64_any_dtype(df['legs1_departureAt']):
            df['legs1_departureAt_dt'] = pd.to_datetime(df['legs1_departureAt'].astype(str), errors='coerce')
            df['num_legs'] = (1 + df['legs1_departureAt_dt'].notna()).astype(np.int8)
            df.drop(columns=['legs1_departureAt_dt'], inplace=True) # Drop temporary column
        else:
            df['num_legs'] = (1 + df['legs1_departureAt'].notna()).astype(np.int8)
    else: 
        df['num_legs'] = np.int8(1) # Assumed 1 if leg1 departure not present
        
    # Ensure num_segments are initialized as int8 from the start
    df['num_segments_leg0'] = np.int8(0)
    df['num_segments_leg1'] = np.int8(0)
    for i in range(4):
        col_l0_seg = f'legs0_segments{i}_departureFrom_airport_iata'
        if col_l0_seg in df.columns: df['num_segments_leg0'] += df[col_l0_seg].notna().astype(np.int8)
        col_l1_seg = f'legs1_segments{i}_departureFrom_airport_iata'
        if col_l1_seg in df.columns: df['num_segments_leg1'] += df[col_l1_seg].notna().astype(np.int8)
    df['total_segments'] = (df['num_segments_leg0'] + df['num_segments_leg1']).astype(np.int8)
    
    df['is_direct_leg0'] = (df['num_segments_leg0'] == 1).astype(np.int8)
    df['is_direct_leg1'] = (df['num_segments_leg1'] == 1).astype(np.int8)
    df['both_direct'] = (df['is_direct_leg0'] & df['is_direct_leg1']).astype(np.int8)

    # --- Duration and Price based features ---
    for dur_col in ['legs0_duration', 'legs1_duration']:
        if dur_col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[dur_col]):
                df[dur_col] = pd.to_numeric(df[dur_col].astype(str), errors='coerce').fillna(0)
            else: 
                df[dur_col] = df[dur_col].fillna(0)
        else: 
            df[dur_col] = 0 
    df['legs0_duration'] = df['legs0_duration'].astype(np.float32)
    df['legs1_duration'] = df['legs1_duration'].astype(np.float32)
    df['total_flight_duration'] = (df['legs0_duration'] + df['legs1_duration']).astype(np.float32)
    
    if 'totalPrice' in df.columns and 'total_flight_duration' in df.columns:
        df['price_per_minute'] = (df['totalPrice'] / (df['total_flight_duration'] + 1e-6)).fillna(0).astype(np.float32)
    else:
        df['price_per_minute'] = np.float32(0.0)
            
    if 'totalPrice' in df.columns and 'taxes' in df.columns:
        # Ensure taxes is numeric and fillna before division
        df_taxes_numeric = pd.to_numeric(df['taxes'], errors='coerce').fillna(0)
        df_totalPrice_numeric = pd.to_numeric(df['totalPrice'], errors='coerce').fillna(0)
        
        df['tax_ratio'] = (df_taxes_numeric / (df_totalPrice_numeric + 1e-6)).fillna(0).astype(np.float32)
        df['price_per_tax'] = (df_totalPrice_numeric / (df_taxes_numeric + 1e-6)).fillna(0).astype(np.float32)
        del df_taxes_numeric, df_totalPrice_numeric
    else:
        df['tax_ratio'] = np.float32(0.0)
        df['price_per_tax'] = np.float32(0.0)
        
    if 'pricingInfo_isAccessTP' in df.columns: 
        df['is_compliant'] = pd.to_numeric(df['pricingInfo_isAccessTP'], errors='coerce').fillna(0).astype(np.int8)
    else: 
        df['is_compliant'] = np.int8(-1)
    
    # --- Baggage features ---
    df['baggage_leg0_qty'] = np.int8(0)
    df['baggage_leg0_included'] = np.int8(-1)
    if 'legs0_segments0_baggageAllowance_quantity' in df.columns: 
        df['baggage_leg0_qty'] = pd.to_numeric(df['legs0_segments0_baggageAllowance_quantity'], errors='coerce').fillna(0).astype(np.int8)
        df['baggage_leg0_included'] = (df['baggage_leg0_qty'] > 0).astype(np.int8)
        
    df['baggage_leg1_qty'] = np.int8(0)
    df['baggage_leg1_included'] = np.int8(0) 
    df['baggage_both_legs_included'] = np.int8(-1)

    if 'legs1_segments0_baggageAllowance_quantity' in df.columns:
        df['baggage_leg1_qty'] = pd.to_numeric(df['legs1_segments0_baggageAllowance_quantity'], errors='coerce').fillna(0).astype(np.int8)
        df['baggage_leg1_included'] = (df['baggage_leg1_qty'] > 0).astype(np.int8)
        # Check if baggage_leg0_included Series exists and is not entirely -1
        if 'baggage_leg0_included' in df.columns and not (df['baggage_leg0_included'] == -1).all():
             df['baggage_both_legs_included'] = (df['baggage_leg0_included'] & df['baggage_leg1_included']).astype(np.int8)
    elif 'baggage_leg0_included' in df.columns and not (df['baggage_leg0_included'] == -1).all():
        df['baggage_both_legs_included'] = df['baggage_leg0_included'].astype(np.int8)
    df['baggage_total_qty'] = (df['baggage_leg0_qty'] + df['baggage_leg1_qty']).astype(np.int8)

    # --- Rules based features (cancel, exchange, fees) ---
    df['free_cancel'] = np.int8(-1); df['free_exchange'] = np.int8(-1)
    if 'miniRules0_monetaryAmount' in df.columns and 'miniRules0_percentage' in df.columns:
        monetary0 = pd.to_numeric(df['miniRules0_monetaryAmount'], errors='coerce').fillna(1)
        percent0 = pd.to_numeric(df['miniRules0_percentage'], errors='coerce').fillna(1)
        df['free_cancel'] = ((monetary0 == 0) & (percent0 == 0)).astype(np.int8)
        del monetary0, percent0
    if 'miniRules1_monetaryAmount' in df.columns and 'miniRules1_percentage' in df.columns:
        monetary1 = pd.to_numeric(df['miniRules1_monetaryAmount'], errors='coerce').fillna(1)
        percent1 = pd.to_numeric(df['miniRules1_percentage'], errors='coerce').fillna(1)
        df['free_exchange'] = ((monetary1 == 0) & (percent1 == 0)).astype(np.int8)
        del monetary1, percent1

    df['total_fees'] = np.float32(0.0)
    if 'miniRules0_monetaryAmount' in df.columns:
        df['total_fees'] += pd.to_numeric(df['miniRules0_monetaryAmount'], errors='coerce').fillna(0)
    if 'miniRules1_monetaryAmount' in df.columns:
        df['total_fees'] += pd.to_numeric(df['miniRules1_monetaryAmount'], errors='coerce').fillna(0)
    df['total_fees'] = df['total_fees'].astype(np.float32) # ensure it's float32 before division
    df['has_fees'] = (df['total_fees'] > 0).astype(np.int8)

    if 'totalPrice' in df.columns:
        df_totalPrice_numeric = pd.to_numeric(df['totalPrice'], errors='coerce').fillna(0)
        df['fee_rate'] = (df['total_fees'] / (df_totalPrice_numeric + 1e-6)).fillna(0).astype(np.float32)
        del df_totalPrice_numeric
    else: 
        df['fee_rate'] = np.float32(0.0)
        
    # --- Cabin Class features ---
    df['legs0_segments0_cabinClass_num'] = np.nan
    df['legs1_segments0_cabinClass_num'] = np.nan
    if 'legs0_segments0_cabinClass' in df.columns:
        df['legs0_segments0_cabinClass_num'] = pd.to_numeric(df['legs0_segments0_cabinClass'], errors='coerce')
    if 'legs1_segments0_cabinClass' in df.columns:
        df['legs1_segments0_cabinClass_num'] = pd.to_numeric(df['legs1_segments0_cabinClass'], errors='coerce')

    # Fill NaN with a value like -1 before mean if both are NaN for a row, or use .mean() default behavior
    avg_cabin_class_temp = df[['legs0_segments0_cabinClass_num', 'legs1_segments0_cabinClass_num']].mean(axis=1)
    df['avg_cabin_class'] = avg_cabin_class_temp.fillna(-1).astype(np.float32)
    
    # When calculating diff, fill NaNs with the average if available, otherwise 0 or -1 if avg is also -1
    l0_cabin_filled = df['legs0_segments0_cabinClass_num'].fillna(avg_cabin_class_temp).fillna(-1)
    l1_cabin_filled = df['legs1_segments0_cabinClass_num'].fillna(avg_cabin_class_temp).fillna(-1)
    df['cabin_class_diff'] = (l0_cabin_filled - l1_cabin_filled).astype(np.float32)
    
    df.drop(columns=['legs0_segments0_cabinClass_num', 'legs1_segments0_cabinClass_num'], inplace=True, errors='ignore')
    del avg_cabin_class_temp, l0_cabin_filled, l1_cabin_filled

    # --- Binary user/trip related features ---
    df['frequentFlyer_binary'] = np.int8(0)
    df['is_vip_freq'] = np.int8(0) 
    if 'isVip' in df.columns:
        is_vip_col_temp = df['isVip'].fillna(0)
        if pd.api.types.is_bool_dtype(is_vip_col_temp): is_vip_col_temp = is_vip_col_temp.astype(int)
        else: is_vip_col_temp = pd.to_numeric(is_vip_col_temp, errors='coerce').fillna(0).astype(int)
        df['is_vip_freq'] = (is_vip_col_temp == 1).astype(np.int8)
        del is_vip_col_temp
        
    if 'frequentFlyer' in df.columns:
        df['frequentFlyer_binary'] = pd.to_numeric(df['frequentFlyer'], errors='coerce').fillna(0).astype(np.int8)
        df['is_vip_freq'] = (df['is_vip_freq'] | (df['frequentFlyer_binary'] == 1)).astype(np.int8)
    
    if 'corporateTariffCode' in df.columns:
        df['has_corporate_tariff'] = (~df['corporateTariffCode'].astype(str).isna() & \
                                     (df['corporateTariffCode'].astype(str) != '') & \
                                     (df['corporateTariffCode'].astype(str).str.upper() != 'NAN') & \
                                     (df['corporateTariffCode'].astype(str).str.upper() != 'MISSING')).astype(np.int8)
    else:
        df['has_corporate_tariff'] = np.int8(-1)
    
    gc.collect()
    print(f"After basic FE. df memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # --- Group-wise features ---
    group_key = 'ranker_id'
    if group_key not in df.columns: return df

    key_numeric_features = []
    for col_candidate in ['totalPrice', 'total_flight_duration', 'booking_lead_days', 'fee_rate', 'total_fees']:
        if col_candidate in df.columns and pd.api.types.is_numeric_dtype(df[col_candidate]):
            if col_candidate == 'booking_lead_days' and (df[col_candidate] == -1.0).all():
                continue
            key_numeric_features.append(col_candidate)
    
    print(f"Processing group-wise features for {'train' if is_train else 'test'} on columns: {key_numeric_features}")
    for col in key_numeric_features:
        print(f"  Calculating group features for {col}...")
        # Ensure source column is float32 for transform operations to maintain precision then downcast
        source_col_float32 = df[col].astype(np.float32)

        df[f'{col}_rank_in_group'] = df.groupby(group_key)[col].rank(method='dense', ascending=True).astype(np.float16)
        df[f'{col}_pct_rank_in_group'] = df.groupby(group_key)[col].rank(method='dense', ascending=True, pct=True).astype(np.float16)
        
        group_min = df.groupby(group_key)[col].transform('min').astype(np.float32)
        df[f'{col}_vs_group_min'] = (source_col_float32 - group_min).astype(np.float16)
        df[f'is_min_{col}_in_group'] = (source_col_float32 == group_min).astype(np.int8) 
        del group_min; gc.collect()
        
        group_mean = df.groupby(group_key)[col].transform('mean').astype(np.float32)
        df[f'{col}_vs_group_mean'] = (source_col_float32 - group_mean).astype(np.float16)
        
        group_std = df.groupby(group_key)[col].transform('std').astype(np.float32).fillna(np.float32(1e-6))
        df[f'{col}_zscore_in_group'] = ((source_col_float32 - group_mean) / group_std).astype(np.float16)
        del group_mean, group_std, source_col_float32; gc.collect()

    if 'totalPrice' in df.columns and 'is_compliant' in df.columns:
        df['price_compliant_temp'] = df['totalPrice'].astype(np.float32) # Ensure float for NaN
        # Ensure is_compliant is numeric for loc
        is_compliant_numeric = pd.to_numeric(df['is_compliant'], errors='coerce').fillna(0)
        df.loc[is_compliant_numeric == 0, 'price_compliant_temp'] = np.nan
        del is_compliant_numeric
        
        min_compliant_price_in_group = df.groupby(group_key)['price_compliant_temp'].transform('min').astype(np.float32)
        df['price_vs_min_compliant_price'] = (df['totalPrice'].astype(np.float32) - min_compliant_price_in_group).astype(np.float16)
        df['price_vs_min_compliant_price'] = df['price_vs_min_compliant_price'].fillna(np.float16(0.0)) 
        df.drop(columns=['price_compliant_temp'], inplace=True)
        del min_compliant_price_in_group; gc.collect()
    
    print(f"After group FE. df memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # --- User/Company Categorical ---
    user_company_cats_loaded = [
        'sex', 'nationality', 'isVip', 
        'corporateTariffCode', 'frequentFlyer',
        'legs0_segments0_cabinClass', 'legs1_segments0_cabinClass'
    ]
    for col in user_company_cats_loaded:
        if col in df.columns:
            # Convert to object first if it's a nullable integer type or bool
            if pd.api.types.is_bool_dtype(df[col]) or \
               (pd.api.types.is_integer_dtype(df[col]) and not pd.api.types.is_categorical_dtype(df[col])):
                df[col] = df[col].astype('object')
            # Or if it's float but represents categories (like cabinClass after to_numeric if it was loaded as float)
            elif pd.api.types.is_float_dtype(df[col]) and col in ['legs0_segments0_cabinClass', 'legs1_segments0_cabinClass']:
                 df[col] = df[col].astype('object')

            df[col] = df[col].fillna('MISSING').astype('category')
    
    binary_cols_loaded = [c for c in ['bySelf', 'isAccess3D'] if c in df.columns]
    for col in binary_cols_loaded: 
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.int8)
    
    print(f"End of FE. Final df memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    df = reduce_mem_usage(df, verbose=False) # Reduce memory one last time within the function
    print(f"End of FE after final reduce_mem_usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    return df

# --- Execution part of Cell 4 (SEPARATE PROCESSING) ---
print("--- Processing TRAIN_DF ---")
print("Initial datetime conversion for train_df...")
train_df_processed = create_initial_datetime_features(train_df.copy())
del train_df; gc.collect()

print("Applying reduce_mem_usage to train_df_processed before main FE...")
train_df_processed = reduce_mem_usage(train_df_processed) 
gc.collect()

print("Creating remaining features for train_df_processed...")
train_df_processed = create_remaining_features(train_df_processed, is_train=True)
gc.collect()

# No need for another reduce_mem_usage here if it's done at the end of create_remaining_features


train_labels = train_df_processed['selected']
train_ids = train_df_processed['Id']
train_ranker_ids = train_df_processed['ranker_id']

raw_datetime_col_names = ['requestDate', 'legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt']
original_categorical_like_cols = [
    'sex', 'nationality', 'isVip', 
    'corporateTariffCode', 'frequentFlyer',
    'legs0_segments0_cabinClass', 'legs1_segments0_cabinClass' 
]
id_cols_and_target = ['Id', 'ranker_id', 'selected', 'profileId', 'companyID', 'searchRoute'] 

excluded_for_X_train = id_cols_and_target + raw_datetime_col_names + \
                       [c for c in original_categorical_like_cols if c in train_df_processed.columns]
train_feature_cols = [col for col in train_df_processed.columns if col not in excluded_for_X_train]


X = train_df_processed[train_feature_cols].copy()
y = train_labels.copy()
print(f"Shape of X_train: {X.shape}"); print(f"X_train memory: {X.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
del train_df_processed; gc.collect()


print("\n--- Processing TEST_DF ---")
print("Initial datetime conversion for test_df...")
test_df_processed = create_initial_datetime_features(test_df.copy())
del test_df; gc.collect()

print("Applying reduce_mem_usage to test_df_processed before main FE...")
test_df_processed = reduce_mem_usage(test_df_processed)
gc.collect()

print("Creating remaining features for test_df_processed...")
test_df_processed = create_remaining_features(test_df_processed, is_train=False)
gc.collect()

# No need for another reduce_mem_usage here

X_test = pd.DataFrame(columns=train_feature_cols, index=test_df_processed.index)
for col in train_feature_cols:
    if col in test_df_processed.columns:
        X_test[col] = test_df_processed[col]
    else:
        # Check dtype of the column in X (train) to decide fill value for X_test
        if X[col].dtype.name.startswith('float') or X[col].dtype.name.startswith('int'):
             X_test[col] = 0 
        else: # Should be category after LabelEncoding, or object if LE hasn't happened yet
             X_test[col] = "MISSING_IN_TEST"

del test_df_processed; gc.collect()

print(f"Shape of X_test: {X_test.shape}"); print(f"X_test memory: {X_test.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\nFinal shapes before LabelEncoding: X_train: {X.shape}, X_test: {X_test.shape}")

# %%
# Cell 5: Label Encoding
# Define potential categorical features based on name patterns or known types
# This list includes originals that were set to .astype('category') and new ones
potential_cat_feature_names = [
    'sex', 'nationality', 'isVip', 'corporateTariffCode', 'frequentFlyer', # These were original .astype('category')
    # Add other features that were created as categories or should be treated as such
    # e.g., airport codes if not already numerical and meant to be categorical
    # 'legs0_segments0_departureFrom_airport_iata' # If it wasn't used to create num_segments and you want it as category
]

categorical_features_for_encoding = []
print("\nIdentifying categorical features for Label Encoding from X.columns...")

for col in X.columns:
    # Heuristic: if dtype is object or category, or if name is in our potential list
    if X[col].dtype.name == 'object' or X[col].dtype.name == 'category' or col in potential_cat_feature_names:
        # Additional check: if it's in potential_cat_feature_names but somehow became numeric due to FE, we might not want to LE it
        # However, if it was explicitly made category in FE, it's fine.
        # The original code converts 'sex', 'nationality', etc., to 'category' in FE.
        
        # Check if column exists in X_test before trying to access its dtype
        if col in X_test.columns and (X_test[col].dtype.name == 'object' or X_test[col].dtype.name == 'category' or col in potential_cat_feature_names):
            pass # It's also object/category in test or a known cat feature
        elif col in X_test.columns and X_test[col].dtype.name not in ['object', 'category'] and col not in potential_cat_feature_names:
            print(f"Skipping LE for {col} as it's numeric in X_test and not in potential_cat_feature_names")
            continue # Skip if it's numeric in test and not explicitly listed as cat

        print(f"Column '{col}' (dtype: {X[col].dtype}) identified as categorical for encoding.")
        categorical_features_for_encoding.append(col)
        le = LabelEncoder()
        
        # Handle missing columns in X_test more robustly during Label Encoding
        if col in X_test.columns:
            # Combine unique values from both train and test for fitting the encoder
            # Ensure consistent handling of NaN/missing values by converting to string
            X_col_str = X[col].astype(str).fillna('MISSING_CAT_VALUE')
            X_test_col_str = X_test[col].astype(str).fillna('MISSING_CAT_VALUE')
            
            combined_col_data = pd.concat([X_col_str, X_test_col_str], axis=0).unique()
            le.fit(combined_col_data)
            
            X[col] = le.transform(X_col_str)
            X_test[col] = le.transform(X_test_col_str)
        else:
            # If column is not in X_test at all, only fit_transform on X
            X_col_str = X[col].astype(str).fillna('MISSING_CAT_VALUE')
            X[col] = le.fit_transform(X_col_str)
            # X_test will not have this column if it wasn't created.
            # If it *should* have been created but was missed, that's an earlier issue.
            # For now, we assume X_test alignment handles this.

print(f"\nCategorical features processed with LabelEncoder: {categorical_features_for_encoding}")

# Ensure all columns are numeric after Label Encoding
print("\nChecking for non-numeric columns after LabelEncoding...")
for col in X.columns:
    if not pd.api.types.is_numeric_dtype(X[col]):
        print(f"Warning: Non-numeric column post-LE in X: {col}, dtype: {X[col].dtype}. Forcing numeric, filling NaNs with -1.")
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-1).astype(np.int32) # Or appropriate type
    if col in X_test.columns and not pd.api.types.is_numeric_dtype(X_test[col]):
        print(f"Warning: Non-numeric column post-LE in X_test: {col}, dtype: {X_test[col].dtype}. Forcing numeric, filling NaNs with -1.")
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(-1).astype(np.int32)

final_features_list = list(X.columns)
print(f"\nFinal features for model ({len(final_features_list)}): {final_features_list}")
print("\nX dtypes after all processing:"); print(X.dtypes.value_counts())
gc.collect()

# %%
# Cell 6: Model Training

import lightgbm as lgb
import matplotlib.pyplot as plt
import gc
import numpy as np # Ensure numpy is imported

# --- GLOBAL SUBSAMPLING FOR FINAL TRAINING ---
GLOBAL_TRAIN_SAMPLE_FRAC = 0.3 
NFOLDS = 5 # Keep as originally intended, subsampling will reduce fold data size

if GLOBAL_TRAIN_SAMPLE_FRAC < 1.0:
    print(f"Applying GLOBAL subsample of {GLOBAL_TRAIN_SAMPLE_FRAC*100}% for K-Fold training.")
    
    # Create a temporary DataFrame for sampling
    # Ensure train_ranker_ids is from the full X, y before any previous HPO subsampling
    # Assuming X, y, train_ranker_ids at this point are the full datasets after FE & LE
    
    unique_rankers_full = train_ranker_ids.unique()
    n_sample_groups = int(len(unique_rankers_full) * GLOBAL_TRAIN_SAMPLE_FRAC)
    
    if n_sample_groups < NFOLDS and len(unique_rankers_full) >= NFOLDS:
        print(f"Warning: Sampled groups ({n_sample_groups}) less than NFOLDS ({NFOLDS}). Adjusting sample size to NFOLDS.")
        n_sample_groups = NFOLDS
    elif n_sample_groups == 0 and len(unique_rankers_full) > 0:
        n_sample_groups = 1 # Should be at least NFOLDS if possible
        if n_sample_groups < NFOLDS:
             print(f"CRITICAL: Not enough groups to sample for {NFOLDS} folds. Using all available {len(unique_rankers_full)} groups.")
             n_sample_groups = len(unique_rankers_full)


    if n_sample_groups > 0 and n_sample_groups >= NFOLDS :
        np.random.seed(42)
        sampled_ranker_ids_global = np.random.choice(unique_rankers_full, size=n_sample_groups, replace=False)
        
        # Get indices from the original full train_ranker_ids Series
        sampled_indices_global = train_ranker_ids[train_ranker_ids.isin(sampled_ranker_ids_global)].index
        
        X_run = X.loc[sampled_indices_global].reset_index(drop=True)
        y_run = y.loc[sampled_indices_global].reset_index(drop=True)
        train_ranker_ids_run = train_ranker_ids.loc[sampled_indices_global].reset_index(drop=True)
        
        del sampled_indices_global, unique_rankers_full, sampled_ranker_ids_global
        gc.collect()
        print(f"  X_run shape after global subsampling: {X_run.shape}")
    else:
        print("  Global subsampling resulted in too few groups or zero groups. Using full data (might cause OOM).")
        X_run = X.copy() # 
        y_run = y.copy()
        train_ranker_ids_run = train_ranker_ids.copy()
else:
    print("Using full data for K-Fold training (GLOBAL_TRAIN_SAMPLE_FRAC = 1.0).")
    X_run = X.copy()
    y_run = y.copy()
    train_ranker_ids_run = train_ranker_ids.copy()


params = {'n_estimators': 1100, 
          'learning_rate': 0.07777978129553888,
          'num_leaves': 55, 'max_depth': 10,
          'min_child_samples': 80,
          'subsample': 0.8,
          'colsample_bytree': 0.6,
          'max_bin': 255,
          'reg_alpha': 3.2096452039244645,
          'reg_lambda': 0.06801980497003189,
          'min_split_gain': 0.14,
          'objective': 'lambdarank',
          'metric': 'ndcg',
          'eval_at': [3],
          'boosting_type': 'gbdt',
          'random_state': 42,
          'n_jobs': -1,
          'verbose': -1,
          'seed': 42}

group_kfold = GroupKFold(n_splits=NFOLDS)

oof_preds_scores = np.zeros(len(X_run)) # Adjusted to X_run size
test_preds_scores = np.zeros(len(X_test))
models = []
fold_hit_rates = []

cat_features_for_lgbm_indices_final = [
    X_run.columns.get_loc(col_name) # Use X_run for locating columns
    for col_name in categorical_features_for_encoding if col_name in X_run.columns
]

if cat_features_for_lgbm_indices_final:
    print(f"Using categorical feature indices for LightGBM: {cat_features_for_lgbm_indices_final}")
    print(f"Corresponding feature names: {[X_run.columns[i] for i in cat_features_for_lgbm_indices_final]}\n")
else:
    print("No categorical features identified for LightGBM native handling.\n")


for fold_, (train_idx, val_idx) in enumerate(group_kfold.split(X_run, y_run, groups=train_ranker_ids_run)):
    print(f"====== Fold {fold_ + 1}/{NFOLDS} ======")
    
    if fold_ > 0:
        gc.collect()

    X_train_fold = X_run.iloc[train_idx]
    y_train_fold = y_run.iloc[train_idx]
    X_val_fold = X_run.iloc[val_idx]
    y_val_fold = y_run.iloc[val_idx]

    print(f"  Train fold shape: {X_train_fold.shape}, Val fold shape: {X_val_fold.shape}")

    current_train_fold_ranker_ids = train_ranker_ids_run.iloc[train_idx]
    current_val_fold_ranker_ids = train_ranker_ids_run.iloc[val_idx]

    train_fold_groups = X_train_fold.groupby(current_train_fold_ranker_ids.values).size().to_list()
    val_fold_groups = X_val_fold.groupby(current_val_fold_ranker_ids.values).size().to_list()

    if not train_fold_groups or 0 in train_fold_groups or not val_fold_groups or 0 in val_fold_groups:
        print(f"Skipping fold {fold_ + 1} due to empty or zero-sized groups.")
        continue

    ranker = lgb.LGBMRanker(**params)
    try:
        print(f"  Starting LightGBM fit for fold {fold_ + 1} with params: {params}")
        ranker.fit(
            X_train_fold, y_train_fold,
            group=train_fold_groups,
            eval_set=[(X_val_fold, y_val_fold)],
            eval_group=[val_fold_groups],
            eval_metric='ndcg',
            callbacks=[lgb.early_stopping(100, verbose=100)],
            categorical_feature=cat_features_for_lgbm_indices_final if cat_features_for_lgbm_indices_final else 'auto'
        )
        print(f"  LightGBM fit completed for fold {fold_ + 1}.")
    except Exception as e:
        print(f"Error during LightGBM fit in fold {fold_ + 1}: {e}")
        # print(f"  X_train_fold mem: {X_train_fold.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        # print(f"  X_val_fold mem: {X_val_fold.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        # Fallback or break if needed
        del X_train_fold, y_train_fold, X_val_fold, y_val_fold
        del current_train_fold_ranker_ids, current_val_fold_ranker_ids
        del train_fold_groups, val_fold_groups
        gc.collect()
        continue 

    models.append(ranker)
    val_fold_scores = ranker.predict(X_val_fold)
    oof_preds_scores[val_idx] = val_fold_scores # Store in subsampled OOF array

    if not X_test.empty:
        print(f"  Predicting on X_test (shape: {X_test.shape}) for fold {fold_ + 1}...")
        current_test_preds = ranker.predict(X_test)
        test_preds_scores += current_test_preds / NFOLDS
        del current_test_preds
        gc.collect()

    val_df_for_metric = pd.DataFrame({
        'ranker_id': current_val_fold_ranker_ids,
        'selected': y_val_fold,
        'score': val_fold_scores
    })
    val_df_for_metric['predicted_rank'] = val_df_for_metric.groupby('ranker_id')['score']\
        .rank(method='first', ascending=False).astype(int)

    fold_hr3 = calculate_hit_rate_at_3(val_df_for_metric)
    fold_hit_rates.append(fold_hr3)
    print(f"Fold {fold_ + 1} HitRate@3: {fold_hr3:.4f}")

    del X_train_fold, y_train_fold, X_val_fold, y_val_fold
    del current_train_fold_ranker_ids, current_val_fold_ranker_ids
    del train_fold_groups, val_fold_groups, ranker, val_fold_scores, val_df_for_metric
    gc.collect()

# Final overall evaluation
if models and len(models) > 0: # Check if at least one model was trained
    # If global subsampling was used, OOF score is on that subsample
    print_oof_source = "subsampled" if GLOBAL_TRAIN_SAMPLE_FRAC < 1.0 else "full"
    
    oof_df_for_metric = pd.DataFrame({
        'ranker_id': train_ranker_ids_run, 
        'selected': y_run,                 
        'score': oof_preds_scores
    })

    oof_df_for_metric['predicted_rank'] = oof_df_for_metric.groupby('ranker_id')['score']\
        .rank(method='first', ascending=False).astype(int)

    overall_oof_hr3 = calculate_hit_rate_at_3(oof_df_for_metric)
    print(f"\nOverall OOF HitRate@3 on {print_oof_source} data (based on {len(models)} trained models): {overall_oof_hr3:.4f}")
    
    if fold_hit_rates: print(f"Mean Fold HitRate@3: {np.mean(fold_hit_rates):.4f}")
    
    print(f"\nParameters used for final folds: {params}")
    print("\nFeature Importances (from last successful model):")
    try:
        lgb.plot_importance(models[-1], figsize=(12, max(18, int(len(X_run.columns)/1.5) if X_run.columns.size > 0 else 18)), 
                            max_num_features=len(X_run.columns) if X_run.columns.size > 0 else 20, 
                            importance_type='gain')
        plt.show()
    except Exception as e:
        print(f"Could not plot feature importance: {e}")
else:
    print("No models were trained successfully.")

# %%
# Use the test_ids_df we saved earlier which has original Id and ranker_id
submission_df = test_ids_df.copy()
submission_df['score'] = test_preds_scores 

submission_df['selected'] = submission_df.groupby('ranker_id')['score'].rank(method='first', ascending=False).astype(int)

# Select only required columns and ensure correct order
submission_df = submission_df[['Id', 'ranker_id', 'selected']]

# Check submission format against sample
print("\nSample Submission:")
print(sample_submission_df.head())
print("\nOur Submission:")
print(submission_df.head())

# Save submission
submission_df.to_parquet('submission.parquet', index=False)
submission_df.to_csv('submission.csv', index=False)
print("\nSubmission file 'submission.parquet' created successfully.")
print(f"Submission shape: {submission_df.shape}")

# Basic validation of submission
# 1. All Ids from test set are present
assert len(submission_df) == len(test_ids_df), "Number of rows doesn't match test set"
assert submission_df['Id'].nunique() == len(test_ids_df['Id'].unique()), "Mismatch in unique Ids"

# 2. Ranks are integers and start from 1
assert submission_df['selected'].min() >= 1, "Ranks should be >= 1"
assert submission_df['selected'].dtype == 'int', "Ranks should be integers"

# 3. Ranks are a valid permutation within each group
def check_rank_permutation(group):
    N = len(group)
    sorted_ranks = sorted(list(group['selected']))
    expected_ranks = list(range(1, N + 1))
    if sorted_ranks != expected_ranks:
        print(f"Invalid rank permutation for ranker_id: {group['ranker_id'].iloc[0]}")
        print(f"Expected: {expected_ranks}, Got: {sorted_ranks}")
        return False
    return True

print("Basic submission validation checks passed (row count, Id uniqueness, rank min value, rank dtype).")

# %%


# %%


# %%


# %%
