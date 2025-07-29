# Simple Ensemble using Rank2Score Method - FlightRank 2025
# Combines multiple baseline submissions using rank-to-score transformation

import pandas as pd
import numpy as np

def rank2score(sr, eps=1e-6):
    """Convert ranks to scores (higher is better)"""
    n = sr.max()
    return 1.0 - (sr - 1) / (n + eps)

def score2rank(s):
    """Convert scores back to ranks"""
    return s.rank(method='first', ascending=False).astype(int)

# Load baseline submissions
# Note: These paths should be updated to actual submission files
submissions = {
    'catboost_baseline': '/kaggle/input/catboost-ranker-baseline-flightrank-2025/submission.csv',  # 0.43916
    'lightgbm_ranker': '/kaggle/input/lightgbm-ranker-ndcg-3/submission.csv',  # 0.42163  
    'ensemble_polars': '/kaggle/input/ensemble-with-polars/submission.csv',  # 0.47635
    'xgboost_polars': '/kaggle/input/xgboost-ranker-with-polars/submission.csv'  # 0.48388
}

print("Loading baseline submissions...")

# For demonstration, create dummy DataFrames if files don't exist
dfs = []
weights = [0.1, 0.1, 0.1, 0.7]  # Higher weight for best performing model

for i, (name, path) in enumerate(submissions.items()):
    try:
        df = pd.read_csv(path)
        print(f"Loaded {name}: {df.shape}")
        dfs.append(df)
    except FileNotFoundError:
        print(f"File not found: {path}")
        # Create a dummy DataFrame for demonstration
        # In practice, you would have actual submission files
        dummy_df = pd.DataFrame({
            'Id': range(1000),
            'ranker_id': np.repeat(range(100), 10),  # 100 groups of 10 items each
            'selected': np.tile(range(1, 11), 100)   # Ranks 1-10 repeated
        })
        dfs.append(dummy_df)
        print(f"Created dummy data for {name}")

if len(dfs) != 4:
    raise ValueError("Need exactly 4 submission files")

print("\n--- Converting Ranks to Scores ---")

# Convert each submission from ranks to scores
score_frames = []
for i, df in enumerate(dfs):
    tmp = df[['Id', 'ranker_id', 'selected']].copy()
    
    # Convert ranks to scores within each group
    tmp['score'] = tmp.groupby('ranker_id')['selected'].transform(rank2score)
    
    # Rename score column for merging
    score_frame = tmp[['Id', 'ranker_id', 'score']].rename(columns={'score': f'score_{i}'})
    score_frames.append(score_frame)
    
    print(f"Model {i}: Score range [{tmp['score'].min():.4f}, {tmp['score'].max():.4f}]")

print("\n--- Merging and Blending Scores ---")

# Merge all score DataFrames
merged = score_frames[0]
for i in range(1, 4):
    merged = merged.merge(score_frames[i], on=['Id', 'ranker_id'], how='left')

print(f"Merged DataFrame shape: {merged.shape}")

# Apply weighted blending
score_cols = [f'score_{i}' for i in range(4)]
w = pd.Series(weights, index=score_cols)

print(f"Blending weights: {dict(zip(score_cols, weights))}")

# Calculate weighted average score
merged['score_mean'] = (merged[score_cols] * w).sum(axis=1) / w.sum()

print(f"Blended score range: [{merged['score_mean'].min():.4f}, {merged['score_mean'].max():.4f}]")

print("\n--- Converting Back to Ranks ---")

# Convert blended scores back to ranks within each group
merged['selected'] = merged.groupby('ranker_id')['score_mean'].transform(score2rank)

# Create final submission
submission = merged[['Id', 'ranker_id', 'selected']].copy()

# Verify ranking integrity
print("Verifying ranking integrity...")
rank_check = submission.groupby('ranker_id')['selected'].apply(
    lambda x: sorted(x.tolist()) == list(range(1, len(x)+1))
)

if rank_check.all():
    print("✅ All rankings are valid")
else:
    invalid_groups = rank_check[~rank_check].index.tolist()
    print(f"❌ Invalid rankings found in {len(invalid_groups)} groups: {invalid_groups[:5]}...")
    
    # Fix invalid rankings
    print("Fixing invalid rankings...")
    submission['selected'] = submission.groupby('ranker_id')['score_mean'].rank(
        method='first', ascending=False
    ).astype(int)

# Display submission statistics
print(f"\n--- Submission Statistics ---")
print(f"Total rows: {len(submission):,}")
print(f"Unique groups: {submission['ranker_id'].nunique():,}")
print(f"Rank distribution:")
print(submission['selected'].value_counts().sort_index().head(10))

# Check for any remaining issues
group_sizes = submission.groupby('ranker_id').size()
print(f"Group size range: [{group_sizes.min()}, {group_sizes.max()}]")
print(f"Groups with size != expected: {(group_sizes != group_sizes.mode()[0]).sum()}")

# Save submission
output_file = 'submission_simple_ensemble_rank2score.csv'
submission.to_csv(output_file, index=False, float_format='%.0f')

print(f"\n✅ Submission saved to: {output_file}")
print(f"Shape: {submission.shape}")
print("\nFirst few rows:")
print(submission.head(10))

# Additional validation
print(f"\n--- Final Validation ---")
print(f"All columns present: {set(submission.columns) == {'Id', 'ranker_id', 'selected'}}")
print(f"No missing values: {submission.isnull().sum().sum() == 0}")
print(f"Selected values are integers: {submission['selected'].dtype == 'int64'}")
print(f"Selected values in valid range: {submission['selected'].min()} to {submission['selected'].max()}")