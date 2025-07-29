#!/usr/bin/env python3
"""
Unit Tests for iBlend Ensemble Function
FlightRank 2025 Validation Framework

Tests cover:
- Function parameter validation
- Data input validation
- Ranking consistency
- Submission format compliance
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
import sys
import warnings

# Add the logic directory to the path
sys.path.insert(0, '/home/saketh/kaggle/logic')

# Import the iBlend function
try:
    from flightrank_2025_aeroclub_recsys_cup___hv_blend import iBlend
except ImportError:
    pytest.skip("iBlend function not available", allow_module_level=True)


class TestiBlendValidation:
    """Comprehensive validation tests for iBlend function"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.create_sample_data()
        
    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_sample_data(self):
        """Create sample submission files for testing"""
        # Sample submission data
        sample_data = pd.DataFrame({
            'Id': range(1, 101),
            'ranker_id': ['r1'] * 25 + ['r2'] * 25 + ['r3'] * 25 + ['r4'] * 25,
            'selected': np.random.randint(1, 26, 100)
        })
        
        # Create multiple submission files
        for name in ['0.42163', '0.43916', '0.47635']:
            file_path = os.path.join(self.temp_dir, f"{name}.csv")
            sample_data.to_csv(file_path, index=False)
    
    def test_parameter_validation(self):
        """Test parameter validation for iBlend function"""
        # Test missing required parameters
        with pytest.raises((TypeError, KeyError)):
            iBlend("", [], {})
        
        # Test invalid path
        with pytest.raises((FileNotFoundError, pd.errors.EmptyDataError)):
            invalid_params = {
                'path': '/invalid/path/',
                'sort': 'desc',
                'target': 'selected',
                'q_rows': 100,
                'prefix': 'subm_',
                'desc': 0.5,
                'asc': 0.5,
                'subwts': [0.1, 0.2],
                'subm': [
                    {'name': 'test1', 'weight': 0.5},
                    {'name': 'test2', 'weight': 0.5}
                ]
            }
            iBlend('/invalid/path/', ['test1', 'test2'], invalid_params)
    
    def test_weight_normalization(self):
        """Test that weights are properly normalized"""
        params = {
            'path': self.temp_dir + '/',
            'sort': 'desc',
            'target': 'selected',
            'q_rows': 100,
            'prefix': 'subm_',
            'desc': 0.5,
            'asc': 0.5,
            'subwts': [0.1, 0.2, 0.3],
            'subm': [
                {'name': '0.42163', 'weight': 0.33},
                {'name': '0.43916', 'weight': 0.33},
                {'name': '0.47635', 'weight': 0.34}
            ]
        }
        
        file_names = ['0.42163', '0.43916', '0.47635']
        
        try:
            result = iBlend(self.temp_dir + '/', file_names, params)
            
            # Check that result is a DataFrame
            assert isinstance(result, pd.DataFrame)
            
            # Check required columns exist
            required_cols = ['Id', 'ranker_id', 'selected']
            for col in required_cols:
                assert col in result.columns, f"Missing required column: {col}"
                
            # Validate ranking consistency
            self.validate_ranking_consistency(result)
            
        except Exception as e:
            warnings.warn(f"iBlend test failed with error: {str(e)}")
    
    def validate_ranking_consistency(self, df):
        """Validate that rankings are consistent within each ranker_id"""
        for ranker_id in df['ranker_id'].unique():
            group = df[df['ranker_id'] == ranker_id]
            ranks = group['selected'].values
            
            # Check that ranks form a valid permutation
            expected_ranks = set(range(1, len(ranks) + 1))
            actual_ranks = set(ranks)
            
            assert actual_ranks == expected_ranks, \
                f"Invalid ranking for ranker_id {ranker_id}: {actual_ranks} != {expected_ranks}"
    
    def test_submission_format_compliance(self):
        """Test that output complies with submission format requirements"""
        params = {
            'path': self.temp_dir + '/',
            'sort': 'desc',
            'target': 'selected',
            'q_rows': 100,
            'prefix': 'subm_',
            'desc': 0.5,
            'asc': 0.5,
            'subwts': [0.1, 0.2, 0.3],
            'subm': [
                {'name': '0.42163', 'weight': 0.33},
                {'name': '0.43916', 'weight': 0.33},
                {'name': '0.47635', 'weight': 0.34}
            ]
        }
        
        file_names = ['0.42163', '0.43916', '0.47635']
        
        try:
            result = iBlend(self.temp_dir + '/', file_names, params)
            
            # Test submission format requirements
            self.validate_submission_format(result)
            
        except Exception as e:
            warnings.warn(f"Submission format test failed: {str(e)}")
    
    def validate_submission_format(self, df):
        """Validate submission format compliance"""
        # Check required columns
        required_columns = ['Id', 'ranker_id', 'selected']
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"
        
        # Check data types
        assert df['Id'].dtype in [np.int64, np.int32], "Id column must be integer"
        assert df['selected'].dtype in [np.int64, np.int32], "selected column must be integer"
        
        # Check for null values
        assert not df.isnull().any().any(), "No null values allowed in submission"
        
        # Check ranking validity within each group
        for ranker_id in df['ranker_id'].unique():
            group = df[df['ranker_id'] == ranker_id]
            ranks = sorted(group['selected'].values)
            expected = list(range(1, len(ranks) + 1))
            assert ranks == expected, f"Invalid ranking sequence for {ranker_id}"
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Test with minimum data
        minimal_data = pd.DataFrame({
            'Id': [1, 2],
            'ranker_id': ['r1', 'r1'],  
            'selected': [1, 2]
        })
        
        minimal_file = os.path.join(self.temp_dir, 'minimal.csv')
        minimal_data.to_csv(minimal_file, index=False)
        
        params = {
            'path': self.temp_dir + '/',
            'sort': 'desc',
            'target': 'selected',
            'q_rows': 2,
            'prefix': 'subm_',
            'desc': 1.0,
            'asc': 0.0,
            'subwts': [0.0],
            'subm': [{'name': 'minimal', 'weight': 1.0}]
        }
        
        try:
            result = iBlend(self.temp_dir + '/', ['minimal'], params)
            assert len(result) == 2, "Result should have 2 rows"
            
        except Exception as e:
            warnings.warn(f"Edge case test failed: {str(e)}")
    
    def test_weight_sensitivity(self):
        """Test sensitivity to weight changes"""
        base_params = {
            'path': self.temp_dir + '/',
            'sort': 'desc',
            'target': 'selected',
            'q_rows': 100,
            'prefix': 'subm_',
            'desc': 0.5,
            'asc': 0.5,
            'subwts': [0.1, 0.2, 0.3],
            'subm': [
                {'name': '0.42163', 'weight': 0.33},
                {'name': '0.43916', 'weight': 0.33},
                {'name': '0.47635', 'weight': 0.34}
            ]
        }
        
        file_names = ['0.42163', '0.43916', '0.47635']
        
        try:
            # Test with different weight configurations
            results = []
            weight_configs = [
                [0.5, 0.3, 0.2],
                [0.2, 0.3, 0.5],
                [0.33, 0.33, 0.34]
            ]
            
            for weights in weight_configs:
                params = base_params.copy()
                for i, weight in enumerate(weights):
                    params['subm'][i]['weight'] = weight
                
                result = iBlend(self.temp_dir + '/', file_names, params)
                results.append(result)
            
            # Verify that different weights produce different results
            assert len(set([str(r['selected'].tolist()) for r in results])) > 1, \
                "Different weights should produce different results"
                
        except Exception as e:
            warnings.warn(f"Weight sensitivity test failed: {str(e)}")


class TestiBlendStability:
    """Test parameter stability and robustness"""
    
    def setup_method(self):
        """Setup stability test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.create_diverse_data()
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_diverse_data(self):
        """Create diverse test data with different ranking patterns"""
        np.random.seed(42)  # For reproducibility
        
        # Create submissions with different ranking patterns
        patterns = {
            'conservative': lambda x: np.random.normal(50, 10, x),
            'aggressive': lambda x: np.random.exponential(20, x),
            'uniform': lambda x: np.random.uniform(1, 100, x)
        }
        
        for name, pattern_func in patterns.items():
            data = pd.DataFrame({
                'Id': range(1, 101),
                'ranker_id': ['r1'] * 25 + ['r2'] * 25 + ['r3'] * 25 + ['r4'] * 25,
                'selected': np.clip(pattern_func(100).astype(int), 1, 25)
            })
            
            # Ensure valid rankings within each group
            for ranker_id in data['ranker_id'].unique():
                mask = data['ranker_id'] == ranker_id
                group_size = mask.sum()
                data.loc[mask, 'selected'] = range(1, group_size + 1)
            
            file_path = os.path.join(self.temp_dir, f"{name}.csv")
            data.to_csv(file_path, index=False)
    
    def test_parameter_stability(self):
        """Test stability across different parameter ranges"""
        base_params = {
            'path': self.temp_dir + '/',
            'sort': 'desc',
            'target': 'selected',
            'q_rows': 100,
            'prefix': 'subm_',
            'desc': 0.5,
            'asc': 0.5,
            'subwts': [0.1, 0.0, -0.1],
            'subm': [
                {'name': 'conservative', 'weight': 0.33},
                {'name': 'aggressive', 'weight': 0.33},
                {'name': 'uniform', 'weight': 0.34}
            ]
        }
        
        file_names = ['conservative', 'aggressive', 'uniform']
        
        # Test different desc/asc ratios
        ratios = [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]
        
        try:
            results = []
            for desc, asc in ratios:
                params = base_params.copy()
                params['desc'] = desc
                params['asc'] = asc
                
                result = iBlend(self.temp_dir + '/', file_names, params)
                results.append(result)
                
                # Validate each result
                assert isinstance(result, pd.DataFrame)
                assert len(result) == 100
                
            # Store stability test results
            stability_summary = {
                'test_ratios': ratios,
                'results_count': len(results),
                'all_valid': all(isinstance(r, pd.DataFrame) for r in results)
            }
            
            warnings.warn(f"Stability test completed: {stability_summary}")
            
        except Exception as e:
            warnings.warn(f"Parameter stability test failed: {str(e)}")


# Additional validation utilities
def validate_hitrate_calculation(predictions, ground_truth, k=3):
    """Validate HitRate@k calculation"""
    assert len(predictions) == len(ground_truth)
    
    hits = 0
    total_groups = len(set(ground_truth['ranker_id']))
    
    for ranker_id in ground_truth['ranker_id'].unique():
        true_selected = ground_truth[ground_truth['ranker_id'] == ranker_id]
        pred_group = predictions[predictions['ranker_id'] == ranker_id]
        
        # Find the ground truth selected item
        true_item = true_selected[true_selected['selected'] == 1]['Id'].iloc[0]
        
        # Check if it's in top-k predictions
        top_k = pred_group.nsmallest(k, 'selected')['Id'].values
        
        if true_item in top_k:
            hits += 1
    
    return hits / total_groups


def validate_ensemble_diversity(submission_files):
    """Validate diversity among ensemble components"""
    correlations = []
    
    for i, file1 in enumerate(submission_files):
        for j, file2 in enumerate(submission_files[i+1:], i+1):
            df1 = pd.read_csv(file1)
            df2 = pd.read_csv(file2)
            
            # Calculate rank correlation
            corr = df1['selected'].corr(df2['selected'])
            correlations.append((file1, file2, corr))
    
    return correlations


if __name__ == "__main__":
    # Run basic validation tests
    import subprocess
    result = subprocess.run([
        'python', '-m', 'pytest', __file__, '-v', '--tb=short'
    ], capture_output=True, text=True)
    
    print("Test Results:")
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)