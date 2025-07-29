#!/usr/bin/env python3
"""
Cross-Validation Framework for FlightRank 2025 Ensemble
Implements multiple CV strategies for ranking problems

Features:
- Group-aware cross-validation (respects ranker_id groups)
- Temporal cross-validation for time-based splits
- Stratified CV for balanced validation
- Parameter stability validation
- Ensemble diversity validation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import ndcg_score
import warnings
from typing import List, Tuple, Dict, Any, Optional
import json
from datetime import datetime
import os


class FlightRankCrossValidator:
    """Cross-validation framework specialized for flight ranking problems"""
    
    def __init__(self, 
                 n_splits: int = 5,
                 random_state: int = 42,
                 validation_size: float = 0.2):
        """
        Initialize the cross-validator
        
        Args:
            n_splits: Number of CV folds
            random_state: Random seed for reproducibility
            validation_size: Fraction of data for validation
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.validation_size = validation_size
        self.cv_results = {}
        
    def group_aware_cv(self, 
                       data: pd.DataFrame,
                       target_col: str = 'selected',
                       group_col: str = 'ranker_id') -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Perform group-aware cross-validation that keeps ranker_id groups intact
        
        Args:
            data: Input DataFrame
            target_col: Target column name
            group_col: Grouping column name
            
        Returns:
            List of (train_idx, val_idx) tuples
        """
        # Get unique groups
        unique_groups = data[group_col].unique()
        
        # Use GroupKFold to split by groups
        group_kfold = GroupKFold(n_splits=self.n_splits)
        
        splits = []
        for train_groups_idx, val_groups_idx in group_kfold.split(
            unique_groups, groups=unique_groups):
            
            train_groups = unique_groups[train_groups_idx]
            val_groups = unique_groups[val_groups_idx]
            
            train_idx = data[data[group_col].isin(train_groups)].index.values
            val_idx = data[data[group_col].isin(val_groups)].index.values
            
            splits.append((train_idx, val_idx))
            
        return splits
    
    def temporal_cv(self, 
                    data: pd.DataFrame,
                    time_col: str = 'requestDate',
                    group_col: str = 'ranker_id') -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Perform temporal cross-validation based on request dates
        
        Args:
            data: Input DataFrame with temporal information
            time_col: Column containing temporal information
            group_col: Grouping column name
            
        Returns:
            List of (train_idx, val_idx) tuples
        """
        if time_col not in data.columns:
            warnings.warn(f"Time column {time_col} not found. Using group-aware CV instead.")
            return self.group_aware_cv(data, group_col=group_col)
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
            data[time_col] = pd.to_datetime(data[time_col], errors='coerce')
        
        # Sort by time and group by ranker_id to maintain group integrity
        data_sorted = data.sort_values([time_col, group_col])
        
        # Get unique groups sorted by earliest request time
        group_times = data_sorted.groupby(group_col)[time_col].min().sort_values()
        unique_groups = group_times.index.values
        
        # Use TimeSeriesSplit-like approach on groups
        n_groups = len(unique_groups)
        fold_size = n_groups // self.n_splits
        
        splits = []
        for i in range(self.n_splits):
            val_start = i * fold_size
            val_end = val_start + fold_size if i < self.n_splits - 1 else n_groups
            
            train_groups = unique_groups[:val_start]
            val_groups = unique_groups[val_start:val_end]
            
            if len(train_groups) == 0:
                continue
                
            train_idx = data[data[group_col].isin(train_groups)].index.values
            val_idx = data[data[group_col].isin(val_groups)].index.values
            
            splits.append((train_idx, val_idx))
            
        return splits
    
    def stratified_group_cv(self, 
                           data: pd.DataFrame,
                           stratify_col: str = 'selected',
                           group_col: str = 'ranker_id') -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Perform stratified cross-validation while respecting group boundaries
        
        Args:
            data: Input DataFrame
            stratify_col: Column to stratify on
            group_col: Grouping column name
            
        Returns:
            List of (train_idx, val_idx) tuples
        """
        # Create group-level stratification
        group_stats = data.groupby(group_col).agg({
            stratify_col: ['sum', 'count']
        }).reset_index()
        
        group_stats.columns = [group_col, 'selected_sum', 'total_count']
        group_stats['selection_rate'] = group_stats['selected_sum'] / group_stats['total_count']
        
        # Bin selection rates for stratification
        group_stats['rate_bin'] = pd.qcut(group_stats['selection_rate'], 
                                         q=min(5, len(group_stats)), 
                                         duplicates='drop')
        
        # Use StratifiedKFold on groups
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, 
                             random_state=self.random_state)
        
        splits = []
        for train_groups_idx, val_groups_idx in skf.split(
            group_stats[group_col], group_stats['rate_bin']):
            
            train_groups = group_stats.iloc[train_groups_idx][group_col].values
            val_groups = group_stats.iloc[val_groups_idx][group_col].values
            
            train_idx = data[data[group_col].isin(train_groups)].index.values
            val_idx = data[data[group_col].isin(val_groups)].index.values
            
            splits.append((train_idx, val_idx))
            
        return splits
    
    def validate_ensemble_parameters(self, 
                                   data: pd.DataFrame,
                                   param_grid: Dict[str, List[Any]],
                                   ensemble_func: callable,
                                   cv_strategy: str = 'group_aware') -> Dict[str, Any]:
        """
        Validate ensemble parameters using cross-validation
        
        Args:
            data: Input DataFrame
            param_grid: Grid of parameters to test
            ensemble_func: Function to evaluate ensemble
            cv_strategy: CV strategy to use
            
        Returns:
            Dictionary with validation results
        """
        # Choose CV strategy
        if cv_strategy == 'group_aware':
            splits = self.group_aware_cv(data)
        elif cv_strategy == 'temporal':
            splits = self.temporal_cv(data)
        elif cv_strategy == 'stratified':
            splits = self.stratified_group_cv(data)
        else:
            raise ValueError(f"Unknown CV strategy: {cv_strategy}")
        
        results = []
        
        # Generate parameter combinations
        import itertools
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for param_combo in itertools.product(*param_values):
            params = dict(zip(param_names, param_combo))
            
            fold_scores = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                try:
                    train_data = data.iloc[train_idx]
                    val_data = data.iloc[val_idx]
                    
                    # Apply ensemble function
                    predictions = ensemble_func(train_data, val_data, params)
                    
                    # Calculate HitRate@3
                    score = self.calculate_hitrate_at_k(val_data, predictions, k=3)
                    fold_scores.append(score)
                    
                except Exception as e:
                    warnings.warn(f"Error in fold {fold_idx} with params {params}: {str(e)}")
                    fold_scores.append(0.0)
            
            results.append({
                'params': params,
                'cv_scores': fold_scores,
                'mean_score': np.mean(fold_scores),
                'std_score': np.std(fold_scores),
                'min_score': np.min(fold_scores),
                'max_score': np.max(fold_scores)
            })
        
        # Sort by mean score
        results.sort(key=lambda x: x['mean_score'], reverse=True)
        
        return {
            'best_params': results[0]['params'] if results else None,
            'best_score': results[0]['mean_score'] if results else 0.0,
            'all_results': results,
            'cv_splits': len(splits)
        }
    
    def calculate_hitrate_at_k(self, 
                              ground_truth: pd.DataFrame, 
                              predictions: pd.DataFrame, 
                              k: int = 3) -> float:
        """
        Calculate HitRate@k metric
        
        Args:
            ground_truth: DataFrame with true selections
            predictions: DataFrame with predicted rankings
            k: Top-k threshold
            
        Returns:
            HitRate@k score
        """
        if 'ranker_id' not in ground_truth.columns or 'ranker_id' not in predictions.columns:
            return 0.0
        
        hits = 0
        total_groups = 0
        
        for ranker_id in ground_truth['ranker_id'].unique():
            # Get ground truth for this group
            gt_group = ground_truth[ground_truth['ranker_id'] == ranker_id]
            pred_group = predictions[predictions['ranker_id'] == ranker_id]
            
            if len(gt_group) == 0 or len(pred_group) == 0:
                continue
            
            # Skip groups with 10 or fewer options (as per competition rules)
            if len(gt_group) <= 10:
                continue
                
            total_groups += 1
            
            # Find the true selected item
            true_selected = gt_group[gt_group['selected'] == 1]
            if len(true_selected) == 0:
                continue
                
            true_id = true_selected['Id'].iloc[0]
            
            # Get top-k predictions
            top_k_pred = pred_group.nsmallest(k, 'selected')
            
            if true_id in top_k_pred['Id'].values:
                hits += 1
        
        return hits / total_groups if total_groups > 0 else 0.0
    
    def stability_analysis(self, 
                          data: pd.DataFrame,
                          ensemble_func: callable,
                          base_params: Dict[str, Any],
                          perturbation_range: float = 0.1,
                          n_perturbations: int = 10) -> Dict[str, Any]:
        """
        Analyze parameter stability by adding noise to parameters
        
        Args:
            data: Input DataFrame
            ensemble_func: Ensemble function to test
            base_params: Base parameter configuration
            perturbation_range: Range of parameter perturbation
            n_perturbations: Number of perturbations to test
            
        Returns:
            Stability analysis results
        """
        base_splits = self.group_aware_cv(data)
        base_scores = []
        
        # Calculate base performance
        for train_idx, val_idx in base_splits:
            train_data = data.iloc[train_idx]
            val_data = data.iloc[val_idx]
            
            try:
                predictions = ensemble_func(train_data, val_data, base_params)
                score = self.calculate_hitrate_at_k(val_data, predictions, k=3)
                base_scores.append(score)
            except Exception as e:
                warnings.warn(f"Base evaluation failed: {str(e)}")
                base_scores.append(0.0)
        
        base_mean = np.mean(base_scores)
        
        # Test perturbations
        perturbation_results = []
        np.random.seed(self.random_state)
        
        for _ in range(n_perturbations):
            perturbed_params = base_params.copy()
            
            # Add noise to numeric parameters
            for key, value in base_params.items():
                if isinstance(value, (int, float)):
                    noise = np.random.normal(0, perturbation_range * abs(value))
                    perturbed_params[key] = value + noise
                elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                    noise = [np.random.normal(0, perturbation_range * abs(x)) for x in value]
                    perturbed_params[key] = [x + n for x, n in zip(value, noise)]
            
            # Evaluate perturbed parameters
            perturbed_scores = []
            for train_idx, val_idx in base_splits:
                train_data = data.iloc[train_idx]
                val_data = data.iloc[val_idx]
                
                try:
                    predictions = ensemble_func(train_data, val_data, perturbed_params)
                    score = self.calculate_hitrate_at_k(val_data, predictions, k=3)
                    perturbed_scores.append(score)
                except Exception as e:
                    perturbed_scores.append(0.0)
            
            perturbation_results.append({
                'params': perturbed_params,
                'scores': perturbed_scores,
                'mean_score': np.mean(perturbed_scores),
                'score_diff': np.mean(perturbed_scores) - base_mean
            })
        
        return {
            'base_score': base_mean,
            'base_std': np.std(base_scores),
            'perturbation_results': perturbation_results,
            'stability_score': np.std([r['score_diff'] for r in perturbation_results]),
            'worst_degradation': min([r['score_diff'] for r in perturbation_results]),
            'best_improvement': max([r['score_diff'] for r in perturbation_results])
        }
    
    def save_validation_report(self, 
                             results: Dict[str, Any], 
                             output_path: str = 'validation_report.json'):
        """
        Save validation results to a JSON report
        
        Args:
            results: Validation results dictionary
            output_path: Path to save the report
        """
        # Add metadata
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'cv_config': {
                'n_splits': self.n_splits,
                'random_state': self.random_state,
                'validation_size': self.validation_size
            },
            'results': results
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', 
                    exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Validation report saved to: {output_path}")


def ensemble_mock_function(train_data, val_data, params):
    """
    Mock ensemble function for testing purposes
    
    Args:
        train_data: Training data
        val_data: Validation data  
        params: Parameters dictionary
        
    Returns:
        DataFrame with predictions
    """
    # Simple mock: return validation data with random rankings
    result = val_data.copy()
    
    for ranker_id in result['ranker_id'].unique():
        mask = result['ranker_id'] == ranker_id
        group_size = mask.sum()
        
        # Create random but valid rankings
        rankings = np.random.permutation(range(1, group_size + 1))
        result.loc[mask, 'selected'] = rankings
    
    return result


if __name__ == "__main__":
    # Example usage and testing
    print("FlightRank Cross-Validation Framework")
    print("=====================================")
    
    # Create sample data for testing
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'Id': range(1, 1001),
        'ranker_id': np.repeat([f'r{i}' for i in range(1, 41)], 25),
        'selected': np.tile(range(1, 26), 40),
        'requestDate': pd.date_range('2025-01-01', periods=1000, freq='H')
    })
    
    # Randomize some selections to create realistic data
    for ranker_id in sample_data['ranker_id'].unique():
        mask = sample_data['ranker_id'] == ranker_id
        selected_idx = np.random.choice(sample_data[mask].index, 1)[0]
        sample_data.loc[mask, 'selected'] = 0
        sample_data.loc[selected_idx, 'selected'] = 1
    
    # Initialize validator
    cv = FlightRankCrossValidator(n_splits=3)
    
    # Test different CV strategies
    print("Testing Group-Aware CV...")
    group_splits = cv.group_aware_cv(sample_data)
    print(f"Created {len(group_splits)} folds")
    
    print("\nTesting Temporal CV...")
    temporal_splits = cv.temporal_cv(sample_data)  
    print(f"Created {len(temporal_splits)} folds")
    
    print("\nTesting Parameter Validation...")
    param_grid = {
        'weight1': [0.3, 0.5, 0.7],
        'weight2': [0.2, 0.3, 0.5]
    }
    
    validation_results = cv.validate_ensemble_parameters(
        sample_data, param_grid, ensemble_mock_function
    )
    
    print(f"Best parameters: {validation_results['best_params']}")
    print(f"Best score: {validation_results['best_score']:.4f}")
    
    print("\nTesting Stability Analysis...")
    base_params = {'weight1': 0.5, 'weight2': 0.3}
    stability_results = cv.stability_analysis(
        sample_data, ensemble_mock_function, base_params
    )
    
    print(f"Base score: {stability_results['base_score']:.4f}")
    print(f"Stability score: {stability_results['stability_score']:.4f}")
    
    # Save report
    all_results = {
        'parameter_validation': validation_results,
        'stability_analysis': stability_results
    }
    
    cv.save_validation_report(all_results, 'validation/cross_validation/cv_report.json')
    print("\nValidation complete! Report saved.")