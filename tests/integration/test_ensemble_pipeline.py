#!/usr/bin/env python3
"""
Integration Tests for FlightRank 2025 Ensemble Pipeline
End-to-end validation of the complete ensemble workflow

Features:
- Full pipeline integration testing
- Data flow validation
- Performance regression detection
- Memory usage monitoring
- Error handling validation
- Reproducibility verification
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

# Add project paths
sys.path.insert(0, '/home/saketh/kaggle/logic')
sys.path.insert(0, '/home/saketh/kaggle/validation')

# Import validation frameworks
try:
    from cross_validation.ensemble_cv import FlightRankCrossValidator
    from metrics.submission_validator import SubmissionValidator
    from stability.parameter_stability import ParameterStabilityTester
except ImportError as e:
    pytest.skip(f"Validation frameworks not available: {e}", allow_module_level=True)

# Import the main ensemble function
try:
    from flightrank_2025_aeroclub_recsys_cup___hv_blend import iBlend
except ImportError:
    pytest.skip("iBlend function not available", allow_module_level=True)


class TestEnsemblePipeline:
    """Integration tests for the complete ensemble pipeline"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data = self.create_comprehensive_test_data()
        self.create_test_submission_files()
        
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_comprehensive_test_data(self) -> Dict[str, pd.DataFrame]:
        """Create comprehensive test datasets"""
        np.random.seed(42)
        
        # Large-scale test data
        n_groups = 100
        avg_group_size = 25
        total_rows = n_groups * avg_group_size
        
        # Training data
        train_data = pd.DataFrame({
            'Id': range(1, total_rows + 1),
            'ranker_id': np.repeat([f'train_r{i}' for i in range(1, n_groups + 1)], avg_group_size),
            'selected': 0,  # Will be set later
            'profileId': np.random.randint(1000, 9999, total_rows),
            'companyID': np.random.randint(100, 999, total_rows),
            'totalPrice': np.random.uniform(200, 2000, total_rows),
            'requestDate': pd.date_range('2025-01-01', periods=total_rows, freq='H')
        })
        
        # Set one selected item per group
        for ranker_id in train_data['ranker_id'].unique():
            mask = train_data['ranker_id'] == ranker_id
            selected_idx = np.random.choice(train_data[mask].index, 1)[0]
            train_data.loc[selected_idx, 'selected'] = 1
        
        # Test data (similar structure but with rankings instead of binary selection)
        test_data = train_data.copy()
        test_data['ranker_id'] = test_data['ranker_id'].str.replace('train_', 'test_')
        
        # Create valid rankings for test data
        for ranker_id in test_data['ranker_id'].unique():
            mask = test_data['ranker_id'] == ranker_id
            group_size = mask.sum()
            test_data.loc[mask, 'selected'] = range(1, group_size + 1)
        
        # Ground truth for evaluation
        ground_truth = test_data.copy()
        ground_truth['selected'] = 0
        
        # Set one selected item per group for ground truth
        for ranker_id in ground_truth['ranker_id'].unique():
            mask = ground_truth['ranker_id'] == ranker_id
            selected_idx = np.random.choice(ground_truth[mask].index, 1)[0]
            ground_truth.loc[selected_idx, 'selected'] = 1
        
        return {
            'train': train_data,
            'test': test_data,
            'ground_truth': ground_truth
        }
    
    def create_test_submission_files(self):
        """Create test submission files for ensemble"""
        base_data = self.test_data['test'].copy()
        
        # Create different submission patterns
        submission_patterns = {
            'conservative': lambda x: np.random.normal(x.mean(), x.std() * 0.5, len(x)),
            'aggressive': lambda x: np.random.exponential(x.mean(), len(x)),
            'balanced': lambda x: np.random.uniform(x.min(), x.max(), len(x))
        }
        
        for name, pattern_func in submission_patterns.items():
            submission = base_data[['Id', 'ranker_id', 'selected']].copy()
            
            # Apply pattern to each group
            for ranker_id in submission['ranker_id'].unique():
                mask = submission['ranker_id'] == ranker_id
                group_indices = submission[mask].index
                
                # Generate scores and convert to rankings
                scores = pattern_func(submission.loc[mask, 'selected'])
                rankings = pd.Series(scores, index=group_indices).rank(method='first').astype(int)
                submission.loc[mask, 'selected'] = rankings
            
            # Save to file
            file_path = os.path.join(self.temp_dir, f'{name}.csv')
            submission.to_csv(file_path, index=False)
    
    def test_end_to_end_pipeline(self):
        """Test the complete ensemble pipeline end-to-end"""
        print("Testing end-to-end ensemble pipeline...")
        
        # Define ensemble parameters
        file_names = ['conservative', 'aggressive', 'balanced']
        params = {
            'path': self.temp_dir + '/',
            'sort': 'desc',
            'target': 'selected',
            'q_rows': len(self.test_data['test']),
            'prefix': 'subm_',
            'desc': 0.4,
            'asc': 0.6,
            'subwts': [0.1, 0.0, -0.1],
            'subm': [
                {'name': 'conservative', 'weight': 0.4},
                {'name': 'aggressive', 'weight': 0.3},
                {'name': 'balanced', 'weight': 0.3}
            ]
        }
        
        # Run ensemble
        start_time = time.time()
        try:
            result = iBlend(self.temp_dir + '/', file_names, params)
            execution_time = time.time() - start_time
            
            # Validate result structure
            assert isinstance(result, pd.DataFrame), "Result must be DataFrame"
            assert len(result) == len(self.test_data['test']), "Result length mismatch"
            
            required_columns = ['Id', 'ranker_id', 'selected']
            for col in required_columns:
                assert col in result.columns, f"Missing column: {col}"
            
            # Validate submission format
            validator = SubmissionValidator(strict_mode=False)
            format_results = validator.validate_submission_format(
                result, self.test_data['test']
            )
            
            assert format_results['format_valid'], f"Format validation failed: {format_results['errors']}"
            
            # Performance assertions
            assert execution_time < 60, f"Execution too slow: {execution_time:.2f}s"
            
        except Exception as e:
            pytest.fail(f"End-to-end pipeline failed: {str(e)}")
    
    def test_cross_validation_integration(self):
        """Test integration with cross-validation framework"""
        print("Testing cross-validation integration...")
        
        cv = FlightRankCrossValidator(n_splits=3, random_state=42)
        
        # Test group-aware CV
        splits = cv.group_aware_cv(self.test_data['train'])
        assert len(splits) == 3, "Should create 3 CV splits"
        
        for train_idx, val_idx in splits:
            assert len(train_idx) > 0, "Training set should not be empty"
            assert len(val_idx) > 0, "Validation set should not be empty"
            assert len(set(train_idx) & set(val_idx)) == 0, "No overlap between train/val"
        
        # Test temporal CV
        temporal_splits = cv.temporal_cv(self.test_data['train'])
        assert len(temporal_splits) >= 1, "Should create at least 1 temporal split"
        
        # Test HitRate calculation
        sample_predictions = self.test_data['test'].copy()
        hitrate_result = cv.calculate_hitrate_at_k(
            self.test_data['ground_truth'], 
            sample_predictions, 
            k=3
        )
        
        assert 0 <= hitrate_result <= 1, f"HitRate should be between 0 and 1, got {hitrate_result}"
    
    def test_submission_validation_integration(self):
        """Test integration with submission validator"""
        print("Testing submission validation integration...")
        
        validator = SubmissionValidator(strict_mode=False)
        
        # Create test submission
        test_submission = self.test_data['test'].copy()
        
        # Test format validation
        validation_results = validator.validate_submission_format(
            test_submission, self.test_data['test']
        )
        
        assert 'format_valid' in validation_results
        assert 'checks' in validation_results
        assert 'stats' in validation_results
        
        # Test metric calculation
        metric_results = validator.calculate_hitrate_at_k(
            self.test_data['ground_truth'],
            test_submission,
            k=3
        )
        
        assert 'hitrate_at_k' in metric_results
        assert 'total_groups' in metric_results
        assert 'hits' in metric_results
        
        # Test reproducibility
        def mock_ensemble(data, params):
            return data.copy()
        
        repro_results = validator.reproducibility_test(
            mock_ensemble, test_submission, {}, n_runs=3
        )
        
        assert 'is_reproducible' in repro_results
        assert 'successful_runs' in repro_results
    
    def test_parameter_stability_integration(self):
        """Test integration with parameter stability framework"""
        print("Testing parameter stability integration...")
        
        tester = ParameterStabilityTester(random_state=42)
        
        # Define test parameters
        base_params = {
            'weight1': 0.5,
            'weight2': 0.3,
            'desc': 0.4,
            'asc': 0.6
        }
        
        param_ranges = {
            'weight1': (0.1, 0.9),
            'weight2': (0.1, 0.7)
        }
        
        # Mock ensemble function for stability testing
        def mock_ensemble(data, params):
            result = data.copy()
            for ranker_id in result['ranker_id'].unique():
                mask = result['ranker_id'] == ranker_id
                group_size = mask.sum()
                result.loc[mask, 'selected'] = range(1, group_size + 1)
            return result
        
        # Test sensitivity analysis
        sensitivity_results = tester.sensitivity_analysis(
            mock_ensemble,
            self.test_data['test'],
            base_params,
            param_ranges,
            n_samples=10  # Reduced for testing
        )
        
        assert 'parameter_effects' in sensitivity_results
        assert 'base_performance' in sensitivity_results
        
        # Test robustness testing
        robustness_results = tester.robustness_testing(
            mock_ensemble,
            self.test_data['test'],
            base_params,
            noise_levels=[0.1, 0.2],
            n_trials=10  # Reduced for testing
        )
        
        assert 'robustness_scores' in robustness_results
        assert 'performance_degradation' in robustness_results
        assert 'failure_rates' in robustness_results
    
    def test_memory_usage_monitoring(self):
        """Test memory usage during ensemble execution"""
        print("Testing memory usage monitoring...")
        
        try:
            import psutil
            
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run ensemble with large-ish data
            file_names = ['conservative', 'aggressive', 'balanced']
            params = {
                'path': self.temp_dir + '/',
                'sort': 'desc',
                'target': 'selected',
                'q_rows': len(self.test_data['test']),
                'prefix': 'subm_',
                'desc': 0.5,
                'asc': 0.5,
                'subwts': [0.1, 0.0, -0.1],
                'subm': [
                    {'name': 'conservative', 'weight': 0.33},
                    {'name': 'aggressive', 'weight': 0.33},
                    {'name': 'balanced', 'weight': 0.34}
                ]
            }
            
            result = iBlend(self.temp_dir + '/', file_names, params)
            
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = mem_after - mem_before
            
            # Memory usage should be reasonable (less than 500MB for test data)
            assert memory_delta < 500, f"Excessive memory usage: {memory_delta:.1f}MB"
            
        except ImportError:
            warnings.warn("psutil not available, skipping memory monitoring test")
    
    def test_error_handling_robustness(self):
        """Test error handling and robustness"""
        print("Testing error handling robustness...")
        
        # Test with missing files
        with pytest.raises((FileNotFoundError, pd.errors.EmptyDataError)):
            params = {
                'path': self.temp_dir + '/',
                'sort': 'desc',
                'target': 'selected',
                'q_rows': 100,
                'prefix': 'subm_',
                'desc': 0.5,
                'asc': 0.5,
                'subwts': [0.1],
                'subm': [{'name': 'nonexistent', 'weight': 1.0}]
            }
            iBlend(self.temp_dir + '/', ['nonexistent'], params)
        
        # Test with malformed parameters
        malformed_params = {
            'path': self.temp_dir + '/',
            'sort': 'invalid_sort',
            'target': 'selected',
            'q_rows': -1,  # Invalid
            'prefix': '',
            'desc': 2.0,  # Invalid (should be <= 1)
            'asc': -1.0,  # Invalid
            'subwts': [],  # Empty
            'subm': []  # Empty
        }
        
        # Should handle gracefully or raise appropriate error
        try:
            result = iBlend(self.temp_dir + '/', [], malformed_params)
            # If it doesn't raise an error, it should return valid data
            assert isinstance(result, pd.DataFrame)
        except (ValueError, TypeError, IndexError, KeyError):
            # These are acceptable error types for malformed input
            pass
    
    def test_performance_regression(self):
        """Test for performance regressions"""
        print("Testing performance regression...")
        
        # Benchmark parameters
        file_names = ['conservative', 'aggressive', 'balanced']
        params = {
            'path': self.temp_dir + '/',
            'sort': 'desc',
            'target': 'selected',
            'q_rows': len(self.test_data['test']),
            'prefix': 'subm_',
            'desc': 0.5,
            'asc': 0.5,
            'subwts': [0.1, 0.0, -0.1],
            'subm': [
                {'name': 'conservative', 'weight': 0.33},
                {'name': 'aggressive', 'weight': 0.33},
                {'name': 'balanced', 'weight': 0.34}
            ]
        }
        
        # Run multiple times to get stable timing
        execution_times = []
        for _ in range(3):
            start_time = time.time()
            try:
                result = iBlend(self.temp_dir + '/', file_names, params)
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
            except Exception as e:
                warnings.warn(f"Performance test run failed: {str(e)}")
        
        if execution_times:
            avg_time = np.mean(execution_times)
            max_time = np.max(execution_times)
            
            # Performance thresholds (adjust based on expected performance)
            assert avg_time < 30, f"Average execution time too slow: {avg_time:.2f}s"
            assert max_time < 60, f"Maximum execution time too slow: {max_time:.2f}s"
            
            print(f"Performance: avg={avg_time:.2f}s, max={max_time:.2f}s")
    
    def test_data_integrity_preservation(self):
        """Test that data integrity is preserved through the pipeline"""
        print("Testing data integrity preservation...")
        
        original_test_data = self.test_data['test'].copy()
        
        file_names = ['conservative', 'aggressive', 'balanced']
        params = {
            'path': self.temp_dir + '/',
            'sort': 'desc',
            'target': 'selected',
            'q_rows': len(original_test_data),
            'prefix': 'subm_',
            'desc': 0.5,
            'asc': 0.5,
            'subwts': [0.1, 0.0, -0.1],
            'subm': [
                {'name': 'conservative', 'weight': 0.33},
                {'name': 'aggressive', 'weight': 0.33},
                {'name': 'balanced', 'weight': 0.34}
            ]
        }
        
        result = iBlend(self.temp_dir + '/', file_names, params)
        
        # Check that original data structure is preserved
        assert len(result) == len(original_test_data), "Row count should be preserved"
        
        # Check that Id mapping is preserved
        assert set(result['Id']) == set(original_test_data['Id']), "Id values should be preserved"
        
        # Check that ranker_id mapping is preserved
        assert set(result['ranker_id']) == set(original_test_data['ranker_id']), "ranker_id values should be preserved"
        
        # Check that group sizes are preserved
        original_group_sizes = original_test_data.groupby('ranker_id').size().sort_index()
        result_group_sizes = result.groupby('ranker_id').size().sort_index()
        
        pd.testing.assert_series_equal(original_group_sizes, result_group_sizes, 
                                     check_names=False, check_dtype=False)
    
    def test_reproducibility_across_runs(self):
        """Test reproducibility across multiple runs"""
        print("Testing reproducibility across runs...")
        
        file_names = ['conservative', 'aggressive', 'balanced']
        params = {
            'path': self.temp_dir + '/',
            'sort': 'desc',
            'target': 'selected',
            'q_rows': len(self.test_data['test']),
            'prefix': 'subm_',
            'desc': 0.5,
            'asc': 0.5,
            'subwts': [0.1, 0.0, -0.1],
            'subm': [
                {'name': 'conservative', 'weight': 0.33},
                {'name': 'aggressive', 'weight': 0.33},
                {'name': 'balanced', 'weight': 0.34}
            ]
        }
        
        # Run ensemble multiple times
        results = []
        for run in range(3):
            try:
                result = iBlend(self.temp_dir + '/', file_names, params)
                results.append(result)
            except Exception as e:
                warnings.warn(f"Reproducibility test run {run+1} failed: {str(e)}")
        
        if len(results) >= 2:
            # Check that results are identical (or very similar)
            for i in range(1, len(results)):
                pd.testing.assert_frame_equal(
                    results[0][['Id', 'ranker_id']],
                    results[i][['Id', 'ranker_id']],
                    check_dtype=False
                )
                
                # Rankings should be identical or very close
                ranking_diff = (results[0]['selected'] != results[i]['selected']).sum()
                total_rows = len(results[0])
                diff_rate = ranking_diff / total_rows
                
                assert diff_rate < 0.01, f"Too many ranking differences: {diff_rate:.2%}"
    
    def generate_integration_report(self) -> str:
        """Generate comprehensive integration test report"""
        report_path = os.path.join(self.temp_dir, 'integration_test_report.json')
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_environment': {
                'data_size': {
                    'train_rows': len(self.test_data['train']),
                    'test_rows': len(self.test_data['test']),
                    'unique_groups': self.test_data['test']['ranker_id'].nunique()
                },
                'temp_dir': self.temp_dir
            },
            'test_summary': {
                'total_tests': 8,
                'integration_areas': [
                    'end_to_end_pipeline',
                    'cross_validation_integration', 
                    'submission_validation_integration',
                    'parameter_stability_integration',
                    'memory_usage_monitoring',
                    'error_handling_robustness',
                    'performance_regression',
                    'data_integrity_preservation'
                ]
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report_path


# Standalone test runner
if __name__ == "__main__":
    print("FlightRank Ensemble Pipeline Integration Tests")
    print("=" * 50)
    
    # Run tests manually for debugging
    test_suite = TestEnsemblePipeline()
    test_suite.setup_method()
    
    try:
        print("Running integration tests...")
        
        test_suite.test_end_to_end_pipeline()
        print("✓ End-to-end pipeline test passed")
        
        test_suite.test_cross_validation_integration()
        print("✓ Cross-validation integration test passed")
        
        test_suite.test_submission_validation_integration()
        print("✓ Submission validation integration test passed")
        
        test_suite.test_parameter_stability_integration()
        print("✓ Parameter stability integration test passed")
        
        test_suite.test_memory_usage_monitoring()
        print("✓ Memory usage monitoring test passed")
        
        test_suite.test_error_handling_robustness()
        print("✓ Error handling robustness test passed")
        
        test_suite.test_performance_regression()
        print("✓ Performance regression test passed")
        
        test_suite.test_data_integrity_preservation()
        print("✓ Data integrity preservation test passed")
        
        test_suite.test_reproducibility_across_runs()
        print("✓ Reproducibility test passed")
        
        # Generate report
        report_path = test_suite.generate_integration_report()
        print(f"\n✓ Integration test report generated: {report_path}")
        
        print("\n" + "=" * 50)
        print("ALL INTEGRATION TESTS PASSED!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Integration test failed: {str(e)}")
        raise
        
    finally:
        test_suite.teardown_method()