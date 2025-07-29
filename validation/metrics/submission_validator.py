#!/usr/bin/env python3
"""
Submission Format Validator for FlightRank 2025
Comprehensive validation for competition submission requirements

Features:
- Submission format compliance checking
- Ranking validity verification
- HitRate@3 metric calculation
- Reproducibility testing
- Performance benchmarking
- Anti-cheating validation
"""

import pandas as pd
import numpy as np
import hashlib
import json
import warnings
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
import os


class SubmissionValidator:
    """Comprehensive submission format and quality validator"""
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize the validator
        
        Args:
            strict_mode: If True, fail on any validation error
        """
        self.strict_mode = strict_mode
        self.validation_results = {}
        self.errors = []
        self.warnings = []
        
    def validate_submission_format(self, 
                                 submission_df: pd.DataFrame,
                                 test_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Validate submission format against competition requirements
        
        Args:
            submission_df: Submission DataFrame
            test_df: Optional test DataFrame for reference
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'format_valid': True,
            'errors': [],
            'warnings': [],
            'checks': {}
        }
        
        # Check 1: Required columns
        required_columns = ['Id', 'ranker_id', 'selected']
        missing_columns = [col for col in required_columns if col not in submission_df.columns]
        
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            results['errors'].append(error_msg)
            results['format_valid'] = False
            if self.strict_mode:
                raise ValueError(error_msg)
        
        results['checks']['required_columns'] = {
            'status': 'PASS' if not missing_columns else 'FAIL',
            'missing': missing_columns
        }
        
        # Check 2: Data types
        expected_types = {
            'Id': [np.int64, np.int32, int],
            'ranker_id': [object, str],
            'selected': [np.int64, np.int32, int]
        }
        
        type_errors = []
        for col, expected in expected_types.items():
            if col in submission_df.columns:
                actual_type = submission_df[col].dtype
                if not any(np.issubdtype(actual_type, exp) for exp in expected):
                    type_errors.append(f"{col}: expected {expected}, got {actual_type}")
        
        if type_errors:
            error_msg = f"Data type errors: {type_errors}"
            results['errors'].append(error_msg)
            results['format_valid'] = False
            
        results['checks']['data_types'] = {
            'status': 'PASS' if not type_errors else 'FAIL',
            'errors': type_errors
        }
        
        # Check 3: No null values
        null_counts = submission_df.isnull().sum()
        null_columns = null_counts[null_counts > 0].to_dict()
        
        if null_columns:
            error_msg = f"Null values found: {null_columns}"
            results['errors'].append(error_msg)
            results['format_valid'] = False
            
        results['checks']['null_values'] = {
            'status': 'PASS' if not null_columns else 'FAIL',
            'null_counts': null_columns
        }
        
        # Check 4: Ranking validity within groups
        ranking_errors = self._validate_rankings(submission_df)
        if ranking_errors:
            results['errors'].extend(ranking_errors)
            results['format_valid'] = False
            
        results['checks']['ranking_validity'] = {
            'status': 'PASS' if not ranking_errors else 'FAIL',
            'errors': ranking_errors[:10]  # Limit to first 10 errors
        }
        
        # Check 5: Row order consistency (if test_df provided)
        if test_df is not None:
            order_valid = self._validate_row_order(submission_df, test_df)
            if not order_valid:
                error_msg = "Row order does not match test.csv"
                results['errors'].append(error_msg)
                results['format_valid'] = False
                
            results['checks']['row_order'] = {
                'status': 'PASS' if order_valid else 'FAIL'
            }
        
        # Check 6: Basic anti-cheating measures
        cheating_warnings = self._detect_potential_cheating(submission_df)
        results['warnings'].extend(cheating_warnings)
        
        results['checks']['anti_cheating'] = {
            'status': 'WARNING' if cheating_warnings else 'PASS',
            'warnings': cheating_warnings
        }
        
        # Summary statistics
        results['stats'] = self._calculate_submission_stats(submission_df)
        
        return results
    
    def _validate_rankings(self, df: pd.DataFrame) -> List[str]:
        """Validate ranking permutations within each group"""
        errors = []
        
        for ranker_id in df['ranker_id'].unique():
            group = df[df['ranker_id'] == ranker_id]
            ranks = sorted(group['selected'].values)
            expected = list(range(1, len(ranks) + 1))
            
            if ranks != expected:
                errors.append(f"Invalid ranking for ranker_id {ranker_id}: "
                            f"expected {expected}, got {sorted(group['selected'].values)}")
                
                if len(errors) >= 100:  # Limit error messages
                    errors.append("... (more ranking errors found)")
                    break
        
        return errors
    
    def _validate_row_order(self, submission_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        """Validate that submission maintains test.csv row order"""
        if len(submission_df) != len(test_df):
            return False
            
        # Check Id column order
        return submission_df['Id'].tolist() == test_df['Id'].tolist()
    
    def _detect_potential_cheating(self, df: pd.DataFrame) -> List[str]:
        """Detect potential cheating patterns"""
        warnings = []
        
        # Check for constant rankings
        constant_rankers = []
        for ranker_id in df['ranker_id'].unique():
            group = df[df['ranker_id'] == ranker_id]
            if len(set(group['selected'])) == 1:
                constant_rankers.append(ranker_id)
        
        if len(constant_rankers) > len(df['ranker_id'].unique()) * 0.1:
            warnings.append(f"Suspiciously high number of constant rankings: {len(constant_rankers)}")
        
        # Check for perfectly sorted rankings across all groups
        perfect_sort_count = 0
        for ranker_id in df['ranker_id'].unique():
            group = df[df['ranker_id'] == ranker_id].sort_values('Id')
            if group['selected'].tolist() == list(range(1, len(group) + 1)):
                perfect_sort_count += 1
        
        if perfect_sort_count > len(df['ranker_id'].unique()) * 0.8:
            warnings.append("Suspiciously high number of perfectly sorted rankings")
        
        return warnings
    
    def _calculate_submission_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate submission statistics"""
        stats = {
            'total_rows': len(df),
            'unique_ranker_ids': df['ranker_id'].nunique(),
            'avg_group_size': len(df) / df['ranker_id'].nunique(),
            'min_group_size': df.groupby('ranker_id').size().min(),
            'max_group_size': df.groupby('ranker_id').size().max(),
            'groups_over_10': (df.groupby('ranker_id').size() > 10).sum()
        }
        
        return stats
    
    def calculate_hitrate_at_k(self, 
                              ground_truth: pd.DataFrame,
                              predictions: pd.DataFrame,
                              k: int = 3,
                              min_group_size: int = 11) -> Dict[str, float]:
        """
        Calculate HitRate@k metric with competition-specific rules
        
        Args:
            ground_truth: Ground truth DataFrame with binary selections
            predictions: Predictions DataFrame with rankings
            k: Top-k threshold (default 3)
            min_group_size: Minimum group size to include (default 11)
            
        Returns:
            Dictionary with metric results
        """
        if 'ranker_id' not in ground_truth.columns or 'ranker_id' not in predictions.columns:
            raise ValueError("Both DataFrames must contain 'ranker_id' column")
        
        hits = 0
        total_groups = 0
        processed_groups = 0
        skipped_groups = 0
        
        group_results = []
        
        for ranker_id in ground_truth['ranker_id'].unique():
            gt_group = ground_truth[ground_truth['ranker_id'] == ranker_id]
            pred_group = predictions[predictions['ranker_id'] == ranker_id]
            
            processed_groups += 1
            
            # Skip groups with insufficient size (competition rule)
            if len(gt_group) < min_group_size:
                skipped_groups += 1
                continue
            
            total_groups += 1
            
            # Find the true selected item
            true_selected = gt_group[gt_group['selected'] == 1]
            if len(true_selected) == 0:
                warnings.warn(f"No selected item found for ranker_id {ranker_id}")
                continue
            
            true_id = true_selected['Id'].iloc[0]
            
            # Get predictions for this group
            if len(pred_group) == 0:
                warnings.warn(f"No predictions found for ranker_id {ranker_id}")
                continue
            
            # Get top-k predictions (lowest ranks)
            top_k_pred = pred_group.nsmallest(k, 'selected')
            hit = true_id in top_k_pred['Id'].values
            
            if hit:
                hits += 1
            
            group_results.append({
                'ranker_id': ranker_id,
                'group_size': len(gt_group),
                'true_id': true_id,
                'predicted_rank': pred_group[pred_group['Id'] == true_id]['selected'].iloc[0] if true_id in pred_group['Id'].values else None,
                'hit': hit
            })
        
        hitrate = hits / total_groups if total_groups > 0 else 0.0
        
        return {
            'hitrate_at_k': hitrate,
            'k': k,
            'hits': hits,
            'total_groups': total_groups,
            'processed_groups': processed_groups,
            'skipped_groups': skipped_groups,
            'min_group_size': min_group_size,
            'group_results': group_results
        }
    
    def reproducibility_test(self, 
                           ensemble_func: callable,
                           test_data: pd.DataFrame,
                           params: Dict[str, Any],
                           n_runs: int = 3) -> Dict[str, Any]:
        """
        Test reproducibility of ensemble function
        
        Args:
            ensemble_func: Function to test
            test_data: Test data
            params: Parameters for the function
            n_runs: Number of runs to test
            
        Returns:
            Reproducibility test results
        """
        results = []
        hashes = []
        
        for run_idx in range(n_runs):
            try:
                start_time = time.time()
                result = ensemble_func(test_data, params)
                end_time = time.time()
                
                # Calculate hash of result for comparison
                result_str = result.to_csv(index=False)
                result_hash = hashlib.md5(result_str.encode()).hexdigest()
                
                results.append({
                    'run': run_idx + 1,
                    'execution_time': end_time - start_time,
                    'result_hash': result_hash,
                    'result_shape': result.shape
                })
                
                hashes.append(result_hash)
                
            except Exception as e:
                results.append({
                    'run': run_idx + 1,
                    'error': str(e),
                    'execution_time': None,
                    'result_hash': None
                })
        
        # Check consistency
        unique_hashes = set(h for h in hashes if h is not None)
        is_reproducible = len(unique_hashes) <= 1
        
        return {
            'is_reproducible': is_reproducible,
            'n_runs': n_runs,
            'successful_runs': len([r for r in results if 'error' not in r]),
            'unique_results': len(unique_hashes),
            'avg_execution_time': np.mean([r['execution_time'] for r in results if r['execution_time'] is not None]),
            'std_execution_time': np.std([r['execution_time'] for r in results if r['execution_time'] is not None]),
            'run_details': results
        }
    
    def performance_benchmark(self, 
                            ensemble_func: callable,
                            test_data: pd.DataFrame,
                            params: Dict[str, Any],
                            n_iterations: int = 5) -> Dict[str, Any]:
        """
        Benchmark ensemble performance
        
        Args:
            ensemble_func: Function to benchmark
            test_data: Test data
            params: Function parameters
            n_iterations: Number of benchmark iterations
            
        Returns:
            Performance benchmark results
        """
        execution_times = []
        memory_usage = []
        
        for i in range(n_iterations):
            try:
                # Monitor memory usage (simplified)
                import psutil
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss / 1024 / 1024  # MB
                
                start_time = time.time()
                result = ensemble_func(test_data, params)
                end_time = time.time()
                
                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                
                execution_times.append(end_time - start_time)
                memory_usage.append(mem_after - mem_before)
                
            except ImportError:
                # psutil not available, skip memory monitoring
                start_time = time.time()
                result = ensemble_func(test_data, params)
                end_time = time.time()
                
                execution_times.append(end_time - start_time)
                
            except Exception as e:
                warnings.warn(f"Benchmark iteration {i+1} failed: {str(e)}")
        
        return {
            'n_iterations': n_iterations,
            'successful_iterations': len(execution_times),
            'avg_execution_time': np.mean(execution_times) if execution_times else None,
            'std_execution_time': np.std(execution_times) if execution_times else None,
            'min_execution_time': np.min(execution_times) if execution_times else None,
            'max_execution_time': np.max(execution_times) if execution_times else None,
            'avg_memory_delta': np.mean(memory_usage) if memory_usage else None,
            'execution_times': execution_times,
            'memory_usage': memory_usage
        }
    
    def generate_validation_report(self, 
                                 submission_df: pd.DataFrame,
                                 ground_truth_df: Optional[pd.DataFrame] = None,
                                 test_df: Optional[pd.DataFrame] = None,
                                 output_path: str = 'validation_report.json') -> Dict[str, Any]:
        """
        Generate comprehensive validation report
        
        Args:
            submission_df: Submission DataFrame
            ground_truth_df: Optional ground truth for metric calculation
            test_df: Optional test DataFrame for format validation
            output_path: Path to save the report
            
        Returns:
            Complete validation report
        """
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'submission_info': {
                'shape': submission_df.shape,
                'columns': list(submission_df.columns)
            }
        }
        
        # Format validation
        print("Validating submission format...")
        format_results = self.validate_submission_format(submission_df, test_df)
        report['format_validation'] = format_results
        
        # Metric calculation
        if ground_truth_df is not None:
            print("Calculating metrics...")
            metric_results = self.calculate_hitrate_at_k(ground_truth_df, submission_df)
            report['metrics'] = metric_results
        
        # Save report
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', 
                    exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Validation report saved to: {output_path}")
        
        # Print summary
        self._print_validation_summary(report)
        
        return report
    
    def _print_validation_summary(self, report: Dict[str, Any]):
        """Print validation summary to console"""
        print("\n" + "="*50)
        print("VALIDATION SUMMARY")
        print("="*50)
        
        # Format validation summary
        format_valid = report['format_validation']['format_valid']
        print(f"Format Valid: {'✓ PASS' if format_valid else '✗ FAIL'}")
        
        if not format_valid:
            print("Errors:")
            for error in report['format_validation']['errors'][:5]:
                print(f"  - {error}")
        
        # Metrics summary
        if 'metrics' in report:
            hitrate = report['metrics']['hitrate_at_k']
            print(f"HitRate@3: {hitrate:.4f}")
            print(f"Groups evaluated: {report['metrics']['total_groups']}")
            print(f"Groups skipped: {report['metrics']['skipped_groups']}")
        
        # Submission stats
        stats = report['format_validation']['stats']
        print(f"Total rows: {stats['total_rows']:,}")
        print(f"Unique groups: {stats['unique_ranker_ids']:,}")
        print(f"Avg group size: {stats['avg_group_size']:.1f}")
        print(f"Groups > 10 items: {stats['groups_over_10']:,}")
        
        print("="*50)


def mock_ensemble_function(data, params):
    """Mock ensemble function for testing"""
    result = data.copy()
    
    # Create valid rankings for each group
    for ranker_id in result['ranker_id'].unique():
        mask = result['ranker_id'] == ranker_id
        group_size = mask.sum()
        result.loc[mask, 'selected'] = range(1, group_size + 1)
    
    return result


if __name__ == "__main__":
    # Example usage and testing
    print("FlightRank Submission Validator")
    print("===============================")
    
    # Create sample data
    np.random.seed(42)
    
    # Sample submission data
    submission_data = pd.DataFrame({
        'Id': range(1, 1001),
        'ranker_id': np.repeat([f'r{i}' for i in range(1, 41)], 25),
        'selected': np.tile(range(1, 26), 40)
    })
    
    # Sample ground truth
    ground_truth = submission_data.copy()
    ground_truth['selected'] = 0
    
    # Set one item as selected per group
    for ranker_id in ground_truth['ranker_id'].unique():
        mask = ground_truth['ranker_id'] == ranker_id
        selected_idx = np.random.choice(ground_truth[mask].index, 1)[0]
        ground_truth.loc[selected_idx, 'selected'] = 1
    
    # Initialize validator
    validator = SubmissionValidator(strict_mode=False)
    
    # Run validation
    report = validator.generate_validation_report(
        submission_data, 
        ground_truth, 
        output_path='validation/metrics/sample_validation_report.json'
    )
    
    # Test reproducibility
    print("\nTesting reproducibility...")
    repro_results = validator.reproducibility_test(
        mock_ensemble_function, submission_data, {}, n_runs=3
    )
    print(f"Reproducible: {'✓' if repro_results['is_reproducible'] else '✗'}")
    
    # Performance benchmark
    print("\nRunning performance benchmark...")
    perf_results = validator.performance_benchmark(
        mock_ensemble_function, submission_data, {}, n_iterations=3
    )
    print(f"Avg execution time: {perf_results['avg_execution_time']:.4f}s")
    
    print("\nValidation complete!")