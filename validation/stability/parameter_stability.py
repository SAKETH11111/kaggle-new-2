#!/usr/bin/env python3
"""
Parameter Stability Testing Framework for FlightRank 2025
Tests robustness and stability of ensemble parameters

Features:
- Parameter sensitivity analysis
- Robustness testing with noise injection
- Convergence analysis
- Performance degradation detection
- Confidence intervals for parameters
- Monte Carlo stability testing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional, Callable
import warnings
import json
import os
from pathlib import Path
import time


class ParameterStabilityTester:
    """Comprehensive parameter stability testing framework"""
    
    def __init__(self, 
                 random_state: int = 42,
                 confidence_level: float = 0.95):
        """
        Initialize stability tester
        
        Args:
            random_state: Random seed for reproducibility
            confidence_level: Confidence level for intervals
        """
        self.random_state = random_state
        self.confidence_level = confidence_level
        self.results = {}
        np.random.seed(random_state)
        
    def sensitivity_analysis(self, 
                           ensemble_func: Callable,
                           base_data: pd.DataFrame,
                           base_params: Dict[str, Any],
                           param_ranges: Dict[str, Tuple[float, float]],
                           n_samples: int = 50) -> Dict[str, Any]:
        """
        Perform sensitivity analysis by varying parameters
        
        Args:
            ensemble_func: Ensemble function to test
            base_data: Base dataset
            base_params: Base parameter configuration
            param_ranges: Ranges for each parameter to test
            n_samples: Number of samples per parameter
            
        Returns:
            Sensitivity analysis results
        """
        print("Running sensitivity analysis...")
        
        results = {
            'parameter_effects': {},
            'interaction_effects': {},
            'base_performance': None
        }
        
        # Calculate base performance
        try:
            base_result = ensemble_func(base_data, base_params)
            base_score = self._calculate_performance_score(base_result, base_data)
            results['base_performance'] = base_score
        except Exception as e:
            warnings.warn(f"Base performance calculation failed: {str(e)}")
            results['base_performance'] = 0.0
        
        # Test each parameter individually
        for param_name, (min_val, max_val) in param_ranges.items():
            if param_name not in base_params:
                continue
                
            param_scores = []
            param_values = np.linspace(min_val, max_val, n_samples)
            
            for param_val in param_values:
                test_params = base_params.copy()
                
                # Handle different parameter types
                if isinstance(base_params[param_name], list):
                    # For list parameters, scale all elements
                    original_list = base_params[param_name]
                    scale_factor = param_val / np.mean(original_list)
                    test_params[param_name] = [x * scale_factor for x in original_list]
                else:
                    test_params[param_name] = param_val
                
                try:
                    result = ensemble_func(base_data, test_params)
                    score = self._calculate_performance_score(result, base_data)
                    param_scores.append(score)
                except Exception as e:
                    param_scores.append(0.0)
            
            # Calculate sensitivity metrics
            sensitivity_metrics = self._calculate_sensitivity_metrics(
                param_values, param_scores, results['base_performance']
            )
            
            results['parameter_effects'][param_name] = {
                'values': param_values.tolist(),
                'scores': param_scores,
                'sensitivity': sensitivity_metrics
            }
        
        # Test parameter interactions
        results['interaction_effects'] = self._test_parameter_interactions(
            ensemble_func, base_data, base_params, param_ranges, n_samples=20
        )
        
        return results
    
    def _calculate_sensitivity_metrics(self, 
                                     param_values: np.ndarray,
                                     scores: List[float],
                                     base_score: float) -> Dict[str, float]:
        """Calculate sensitivity metrics for a parameter"""
        scores = np.array(scores)
        
        # Remove invalid scores
        valid_mask = ~np.isnan(scores) & (scores != 0.0)
        if not np.any(valid_mask):
            return {'sensitivity': 0.0, 'correlation': 0.0, 'range_effect': 0.0}
        
        valid_values = param_values[valid_mask]
        valid_scores = scores[valid_mask]
        
        # Calculate correlation
        correlation = np.corrcoef(valid_values, valid_scores)[0, 1] if len(valid_scores) > 1 else 0.0
        
        # Calculate range effect
        range_effect = (np.max(valid_scores) - np.min(valid_scores)) / base_score if base_score != 0 else 0.0
        
        # Calculate sensitivity (derivative approximation)
        if len(valid_scores) > 1:
            sensitivity = np.abs(np.gradient(valid_scores, valid_values)).mean()
        else:
            sensitivity = 0.0
        
        return {
            'sensitivity': float(sensitivity),
            'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'range_effect': float(range_effect),
            'stability_score': 1.0 / (1.0 + range_effect)  # Higher is more stable
        }
    
    def _test_parameter_interactions(self, 
                                   ensemble_func: Callable,
                                   base_data: pd.DataFrame,
                                   base_params: Dict[str, Any],
                                   param_ranges: Dict[str, Tuple[float, float]],
                                   n_samples: int = 20) -> Dict[str, Any]:
        """Test interactions between parameter pairs"""
        param_names = list(param_ranges.keys())
        interactions = {}
        
        # Test pairwise interactions
        for i, param1 in enumerate(param_names):
            for j, param2 in enumerate(param_names[i+1:], i+1):
                try:
                    interaction_score = self._calculate_interaction_effect(
                        ensemble_func, base_data, base_params, 
                        param1, param2, param_ranges, n_samples
                    )
                    interactions[f"{param1}_x_{param2}"] = interaction_score
                except Exception as e:
                    warnings.warn(f"Interaction test failed for {param1} x {param2}: {str(e)}")
                    interactions[f"{param1}_x_{param2}"] = 0.0
        
        return interactions
    
    def _calculate_interaction_effect(self, 
                                    ensemble_func: Callable,
                                    base_data: pd.DataFrame,
                                    base_params: Dict[str, Any],
                                    param1: str,
                                    param2: str,
                                    param_ranges: Dict[str, Tuple[float, float]],
                                    n_samples: int) -> float:
        """Calculate interaction effect between two parameters"""
        # Sample parameter combinations
        p1_min, p1_max = param_ranges[param1]
        p2_min, p2_max = param_ranges[param2]
        
        p1_values = np.linspace(p1_min, p1_max, n_samples)
        p2_values = np.linspace(p2_min, p2_max, n_samples)
        
        # Test corners of parameter space
        corners = [
            (p1_min, p2_min), (p1_min, p2_max),
            (p1_max, p2_min), (p1_max, p2_max)
        ]
        
        corner_scores = []
        for p1_val, p2_val in corners:
            test_params = base_params.copy()
            test_params[param1] = p1_val
            test_params[param2] = p2_val
            
            try:
                result = ensemble_func(base_data, test_params)
                score = self._calculate_performance_score(result, base_data)
                corner_scores.append(score)
            except Exception:
                corner_scores.append(0.0)
        
        # Interaction effect is the variance in corner scores
        return float(np.var(corner_scores)) if corner_scores else 0.0
    
    def robustness_testing(self, 
                          ensemble_func: Callable,
                          base_data: pd.DataFrame,
                          base_params: Dict[str, Any],
                          noise_levels: List[float] = [0.05, 0.1, 0.2, 0.3],
                          n_trials: int = 100) -> Dict[str, Any]:
        """
        Test robustness by injecting noise into parameters
        
        Args:
            ensemble_func: Ensemble function to test
            base_data: Base dataset
            base_params: Base parameter configuration
            noise_levels: Levels of noise to inject
            n_trials: Number of trials per noise level
            
        Returns:
            Robustness test results
        """
        print("Running robustness testing...")
        
        results = {
            'noise_levels': noise_levels,
            'robustness_scores': {},
            'performance_degradation': {},
            'failure_rates': {}
        }
        
        # Calculate baseline performance
        try:
            base_result = ensemble_func(base_data, base_params)
            base_score = self._calculate_performance_score(base_result, base_data)
        except Exception as e:
            warnings.warn(f"Baseline calculation failed: {str(e)}")
            base_score = 0.0
        
        for noise_level in noise_levels:
            scores = []
            failures = 0
            
            for trial in range(n_trials):
                # Add noise to parameters
                noisy_params = self._add_parameter_noise(base_params, noise_level)
                
                try:
                    result = ensemble_func(base_data, noisy_params)
                    score = self._calculate_performance_score(result, base_data)
                    scores.append(score)
                except Exception:
                    failures += 1
                    scores.append(0.0)
            
            # Calculate robustness metrics
            valid_scores = [s for s in scores if s > 0]
            
            results['robustness_scores'][noise_level] = {
                'mean': np.mean(valid_scores) if valid_scores else 0.0,
                'std': np.std(valid_scores) if valid_scores else 0.0,
                'min': np.min(valid_scores) if valid_scores else 0.0,
                'max': np.max(valid_scores) if valid_scores else 0.0,
                'percentile_5': np.percentile(valid_scores, 5) if valid_scores else 0.0,
                'percentile_95': np.percentile(valid_scores, 95) if valid_scores else 0.0
            }
            
            # Performance degradation
            if base_score > 0:
                degradation = (base_score - np.mean(valid_scores)) / base_score if valid_scores else 1.0
            else:
                degradation = 1.0
                
            results['performance_degradation'][noise_level] = degradation
            results['failure_rates'][noise_level] = failures / n_trials
        
        return results
    
    def _add_parameter_noise(self, 
                           params: Dict[str, Any], 
                           noise_level: float) -> Dict[str, Any]:
        """Add Gaussian noise to parameters"""
        noisy_params = params.copy()
        
        for key, value in params.items():
            if isinstance(value, (int, float)):
                noise = np.random.normal(0, noise_level * abs(value))
                noisy_params[key] = value + noise
            elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                noise = [np.random.normal(0, noise_level * abs(x)) for x in value]
                noisy_params[key] = [x + n for x, n in zip(value, noise)]
        
        return noisy_params
    
    def convergence_analysis(self, 
                           ensemble_func: Callable,
                           base_data: pd.DataFrame,
                           param_optimization_history: List[Dict[str, Any]],
                           window_size: int = 10) -> Dict[str, Any]:
        """
        Analyze parameter convergence over optimization history
        
        Args:
            ensemble_func: Ensemble function
            base_data: Base dataset
            param_optimization_history: List of parameter configurations
            window_size: Window size for moving averages
            
        Returns:
            Convergence analysis results
        """
        print("Running convergence analysis...")
        
        if len(param_optimization_history) < 2:
            return {'error': 'Insufficient optimization history'}
        
        # Calculate performance for each configuration
        performances = []
        for i, params in enumerate(param_optimization_history):
            try:
                result = ensemble_func(base_data, params)
                score = self._calculate_performance_score(result, base_data)
                performances.append(score)
            except Exception:
                performances.append(0.0)
        
        # Calculate convergence metrics
        results = {
            'performances': performances,
            'convergence_metrics': {},
            'stability_windows': []
        }
        
        # Moving average convergence
        if len(performances) >= window_size:
            moving_avg = np.convolve(performances, np.ones(window_size)/window_size, mode='valid')
            moving_std = []
            
            for i in range(len(moving_avg)):
                window = performances[i:i+window_size]
                moving_std.append(np.std(window))
            
            results['convergence_metrics'] = {
                'final_performance': performances[-1],
                'best_performance': max(performances),
                'convergence_trend': np.polyfit(range(len(moving_avg)), moving_avg, 1)[0],
                'stability_trend': np.polyfit(range(len(moving_std)), moving_std, 1)[0],
                'final_stability': moving_std[-1] if moving_std else 0.0
            }
        
        # Identify stable windows
        for i in range(0, len(performances) - window_size + 1):
            window = performances[i:i+window_size]
            window_std = np.std(window)
            window_mean = np.mean(window)
            
            results['stability_windows'].append({
                'start_idx': i,
                'end_idx': i + window_size - 1,
                'mean_performance': window_mean,
                'std_performance': window_std,
                'coefficient_of_variation': window_std / window_mean if window_mean != 0 else float('inf')
            })
        
        return results
    
    def monte_carlo_stability(self, 
                            ensemble_func: Callable,
                            base_data: pd.DataFrame,
                            base_params: Dict[str, Any],
                            param_distributions: Dict[str, Tuple[str, Dict]],
                            n_samples: int = 1000) -> Dict[str, Any]:
        """
        Monte Carlo stability testing with parameter distributions
        
        Args:
            ensemble_func: Ensemble function
            base_data: Base dataset
            base_params: Base parameters
            param_distributions: Parameter distributions (name, (dist_type, dist_params))
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Monte Carlo stability results
        """
        print("Running Monte Carlo stability testing...")
        
        results = {
            'n_samples': n_samples,
            'parameter_samples': [],
            'performance_samples': [],
            'stability_metrics': {}
        }
        
        # Generate parameter samples
        for _ in range(n_samples):
            sample_params = base_params.copy()
            
            for param_name, (dist_type, dist_params) in param_distributions.items():
                if param_name not in base_params:
                    continue
                
                # Sample from specified distribution
                if dist_type == 'normal':
                    sample_value = np.random.normal(**dist_params)
                elif dist_type == 'uniform':
                    sample_value = np.random.uniform(**dist_params)
                elif dist_type == 'beta':
                    sample_value = np.random.beta(**dist_params)
                else:
                    continue
                
                sample_params[param_name] = sample_value
            
            results['parameter_samples'].append(sample_params)
            
            # Evaluate performance
            try:
                result = ensemble_func(base_data, sample_params)
                score = self._calculate_performance_score(result, base_data)
                results['performance_samples'].append(score)
            except Exception:
                results['performance_samples'].append(0.0)
        
        # Calculate stability metrics
        valid_scores = [s for s in results['performance_samples'] if s > 0]
        
        if valid_scores:
            results['stability_metrics'] = {
                'mean_performance': np.mean(valid_scores),
                'std_performance': np.std(valid_scores),
                'min_performance': np.min(valid_scores),
                'max_performance': np.max(valid_scores),
                'percentile_5': np.percentile(valid_scores, 5),
                'percentile_25': np.percentile(valid_scores, 25),
                'percentile_75': np.percentile(valid_scores, 75),
                'percentile_95': np.percentile(valid_scores, 95),
                'coefficient_of_variation': np.std(valid_scores) / np.mean(valid_scores),
                'success_rate': len(valid_scores) / n_samples
            }
            
            # Confidence intervals
            alpha = 1 - self.confidence_level
            ci_lower = np.percentile(valid_scores, 100 * alpha / 2)
            ci_upper = np.percentile(valid_scores, 100 * (1 - alpha / 2))
            
            results['stability_metrics']['confidence_interval'] = (ci_lower, ci_upper)
        
        return results
    
    def _calculate_performance_score(self, 
                                   result: pd.DataFrame, 
                                   reference_data: pd.DataFrame) -> float:
        """Calculate a performance score for the ensemble result"""
        # Simple performance metric based on ranking validity
        if not isinstance(result, pd.DataFrame):
            return 0.0
        
        if 'ranker_id' not in result.columns or 'selected' not in result.columns:
            return 0.0
        
        # Check ranking validity
        valid_groups = 0
        total_groups = 0
        
        for ranker_id in result['ranker_id'].unique():
            total_groups += 1
            group = result[result['ranker_id'] == ranker_id]
            ranks = sorted(group['selected'].values)
            expected = list(range(1, len(ranks) + 1))
            
            if ranks == expected:
                valid_groups += 1
        
        return valid_groups / total_groups if total_groups > 0 else 0.0
    
    def generate_stability_report(self, 
                                results: Dict[str, Any],
                                output_dir: str = 'validation/stability') -> str:
        """
        Generate comprehensive stability report
        
        Args:
            results: All stability test results
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated report
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create comprehensive report
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'stability_tests': results,
            'summary': self._generate_stability_summary(results)
        }
        
        # Save JSON report
        report_path = os.path.join(output_dir, 'stability_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate visualizations
        self._create_stability_visualizations(results, output_dir)
        
        # Print summary
        self._print_stability_summary(report['summary'])
        
        return report_path
    
    def _generate_stability_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of stability test results"""
        summary = {
            'overall_stability': 'UNKNOWN',
            'critical_parameters': [],
            'recommendations': []
        }
        
        # Analyze sensitivity results
        if 'sensitivity_analysis' in results:
            sens_results = results['sensitivity_analysis']['parameter_effects']
            
            for param, data in sens_results.items():
                sensitivity = data['sensitivity']['sensitivity']
                stability_score = data['sensitivity']['stability_score']
                
                if sensitivity > 0.1:  # Threshold for high sensitivity
                    summary['critical_parameters'].append({
                        'parameter': param,
                        'sensitivity': sensitivity,
                        'stability_score': stability_score
                    })
        
        # Analyze robustness results
        if 'robustness_testing' in results:
            rob_results = results['robustness_testing']
            
            # Check performance degradation at highest noise level
            max_noise = max(rob_results['noise_levels'])
            max_degradation = rob_results['performance_degradation'][max_noise]
            
            if max_degradation > 0.2:  # More than 20% degradation
                summary['recommendations'].append(
                    f"High performance degradation ({max_degradation:.2%}) at {max_noise} noise level"
                )
        
        # Overall stability assessment
        if len(summary['critical_parameters']) == 0:
            summary['overall_stability'] = 'STABLE'
        elif len(summary['critical_parameters']) <= 2:
            summary['overall_stability'] = 'MODERATELY_STABLE'
        else:
            summary['overall_stability'] = 'UNSTABLE'
        
        return summary
    
    def _create_stability_visualizations(self, results: Dict[str, Any], output_dir: str):
        """Create visualizations for stability results"""
        try:
            import matplotlib.pyplot as plt
            
            # Sensitivity analysis plots
            if 'sensitivity_analysis' in results:
                self._plot_sensitivity_analysis(
                    results['sensitivity_analysis'], 
                    os.path.join(output_dir, 'sensitivity_analysis.png')
                )
            
            # Robustness testing plots
            if 'robustness_testing' in results:
                self._plot_robustness_testing(
                    results['robustness_testing'],
                    os.path.join(output_dir, 'robustness_testing.png')
                )
                
        except ImportError:
            warnings.warn("Matplotlib not available, skipping visualizations")
    
    def _plot_sensitivity_analysis(self, results: Dict[str, Any], output_path: str):
        """Plot sensitivity analysis results"""
        param_effects = results['parameter_effects']
        
        if not param_effects:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (param, data) in enumerate(list(param_effects.items())[:4]):
            if i >= 4:
                break
                
            ax = axes[i]
            ax.plot(data['values'], data['scores'], 'b-', alpha=0.7)
            ax.set_title(f'Sensitivity: {param}')
            ax.set_xlabel('Parameter Value')
            ax.set_ylabel('Performance Score')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_robustness_testing(self, results: Dict[str, Any], output_path: str):
        """Plot robustness testing results"""
        noise_levels = results['noise_levels']
        degradations = [results['performance_degradation'][nl] for nl in noise_levels]
        failure_rates = [results['failure_rates'][nl] for nl in noise_levels]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Performance degradation
        ax1.plot(noise_levels, degradations, 'ro-')
        ax1.set_title('Performance Degradation vs Noise Level')
        ax1.set_xlabel('Noise Level')
        ax1.set_ylabel('Performance Degradation')
        ax1.grid(True, alpha=0.3)
        
        # Failure rates
        ax2.plot(noise_levels, failure_rates, 'bo-')
        ax2.set_title('Failure Rate vs Noise Level')
        ax2.set_xlabel('Noise Level')
        ax2.set_ylabel('Failure Rate')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _print_stability_summary(self, summary: Dict[str, Any]):
        """Print stability summary to console"""
        print("\n" + "="*60)
        print("PARAMETER STABILITY SUMMARY")
        print("="*60)
        
        print(f"Overall Stability: {summary['overall_stability']}")
        
        if summary['critical_parameters']:
            print(f"\nCritical Parameters ({len(summary['critical_parameters'])}):")
            for param_info in summary['critical_parameters']:
                print(f"  - {param_info['parameter']}: "
                      f"sensitivity={param_info['sensitivity']:.4f}, "
                      f"stability={param_info['stability_score']:.4f}")
        
        if summary['recommendations']:
            print(f"\nRecommendations:")
            for rec in summary['recommendations']:
                print(f"  - {rec}")
        
        print("="*60)


def mock_ensemble_function(data, params):
    """Mock ensemble function for testing"""
    result = data.copy()
    
    # Apply some parameter-dependent transformations
    weight_factor = params.get('weight1', 1.0)
    noise_level = params.get('noise_param', 0.0)
    
    for ranker_id in result['ranker_id'].unique():
        mask = result['ranker_id'] == ranker_id
        group_size = mask.sum()
        
        # Create rankings with some parameter dependency
        base_ranks = np.arange(1, group_size + 1)
        
        # Add parameter-dependent noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, group_size)
            noisy_ranks = base_ranks + noise
            # Re-rank to maintain valid permutation
            result.loc[mask, 'selected'] = stats.rankdata(noisy_ranks, method='ordinal')
        else:
            result.loc[mask, 'selected'] = base_ranks
    
    return result


if __name__ == "__main__":
    # Example usage and testing
    print("FlightRank Parameter Stability Tester")
    print("=====================================")
    
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'Id': range(1, 1001),
        'ranker_id': np.repeat([f'r{i}' for i in range(1, 41)], 25),
        'selected': np.tile(range(1, 26), 40)
    })
    
    # Initialize stability tester
    tester = ParameterStabilityTester()
    
    # Define test parameters
    base_params = {
        'weight1': 0.5,
        'weight2': 0.3,
        'noise_param': 0.1
    }
    
    param_ranges = {
        'weight1': (0.1, 0.9),
        'weight2': (0.1, 0.7),
        'noise_param': (0.0, 0.3)
    }
    
    # Run stability tests
    print("Running sensitivity analysis...")
    sensitivity_results = tester.sensitivity_analysis(
        mock_ensemble_function, sample_data, base_params, param_ranges
    )
    
    print("Running robustness testing...")
    robustness_results = tester.robustness_testing(
        mock_ensemble_function, sample_data, base_params
    )
    
    print("Running Monte Carlo stability testing...")
    param_distributions = {
        'weight1': ('normal', {'loc': 0.5, 'scale': 0.1}),
        'weight2': ('uniform', {'low': 0.1, 'high': 0.7}),
        'noise_param': ('beta', {'a': 2, 'b': 5})
    }
    
    mc_results = tester.monte_carlo_stability(
        mock_ensemble_function, sample_data, base_params, 
        param_distributions, n_samples=100
    )
    
    # Compile all results
    all_results = {
        'sensitivity_analysis': sensitivity_results,
        'robustness_testing': robustness_results,
        'monte_carlo_stability': mc_results
    }
    
    # Generate report
    report_path = tester.generate_stability_report(all_results)
    print(f"\nStability report generated: {report_path}")
    
    print("\nParameter stability testing complete!")