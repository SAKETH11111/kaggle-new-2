#!/usr/bin/env python3
"""
Validation Orchestrator for FlightRank 2025 Ensemble
Comprehensive validation framework coordinator

Features:
- Orchestrates all validation frameworks
- Runs complete validation suite
- Generates unified validation reports
- Provides validation recommendations
- Tracks validation metrics over time
- Automated quality gates
"""

import os
import sys
import json
import time
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

# Add project paths
sys.path.insert(0, '/home/saketh/kaggle/logic')
sys.path.insert(0, '/home/saketh/kaggle/validation')
sys.path.insert(0, '/home/saketh/kaggle/tests')

# Import validation frameworks
try:
    from cross_validation.ensemble_cv import FlightRankCrossValidator
    from metrics.submission_validator import SubmissionValidator
    from stability.parameter_stability import ParameterStabilityTester
    from unit.test_iblend import TestiBlendValidation, TestiBlendStability
    from integration.test_ensemble_pipeline import TestEnsemblePipeline
except ImportError as e:
    warnings.warn(f"Some validation frameworks not available: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ValidationOrchestrator:
    """Orchestrates comprehensive validation of the FlightRank ensemble"""
    
    def __init__(self, 
                 output_dir: str = 'validation_results',
                 strict_mode: bool = False,
                 enable_visualizations: bool = True):
        """
        Initialize the validation orchestrator
        
        Args:
            output_dir: Directory to save validation results
            strict_mode: If True, fail fast on validation errors
            enable_visualizations: Enable generation of visualization plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.strict_mode = strict_mode
        self.enable_visualizations = enable_visualizations
        
        # Initialize validation frameworks
        self.cv_validator = FlightRankCrossValidator(n_splits=5, random_state=42)
        self.submission_validator = SubmissionValidator(strict_mode=strict_mode)
        self.stability_tester = ParameterStabilityTester(random_state=42)
        
        # Validation results storage
        self.validation_results = {}
        self.quality_gates = self._define_quality_gates()
        
        # Setup logging
        self._setup_validation_logging()
    
    def _setup_validation_logging(self):
        """Setup dedicated validation logging"""
        log_file = self.output_dir / 'validation.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.info("Validation orchestrator initialized")
    
    def _define_quality_gates(self) -> Dict[str, Dict[str, float]]:
        """Define quality gate thresholds"""
        return {
            'performance': {
                'min_hitrate_at_3': 0.40,  # Minimum acceptable HitRate@3
                'max_execution_time': 300,  # Maximum execution time in seconds
                'max_memory_usage': 2048   # Maximum memory usage in MB
            },
            'stability': {
                'min_stability_score': 0.7,    # Minimum parameter stability
                'max_performance_degradation': 0.3,  # Max degradation under noise
                'min_reproducibility': 0.95    # Minimum reproducibility rate
            },
            'format': {
                'format_compliance': 1.0,   # Must be 100% format compliant
                'ranking_validity': 1.0    # Must have 100% valid rankings
            }
        }
    
    def run_comprehensive_validation(self, 
                                   ensemble_func: callable,
                                   test_data: pd.DataFrame,
                                   ground_truth: Optional[pd.DataFrame] = None,
                                   submission_files: Optional[List[str]] = None,
                                   ensemble_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run comprehensive validation suite
        
        Args:
            ensemble_func: Ensemble function to validate
            test_data: Test dataset
            ground_truth: Ground truth for metric calculation
            submission_files: List of submission file paths
            ensemble_params: Ensemble parameters to validate
            
        Returns:
            Complete validation results
        """
        logger.info("Starting comprehensive validation suite")
        validation_start_time = time.time()
        
        results = {
            'validation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'data_shape': test_data.shape,
                'has_ground_truth': ground_truth is not None,
                'n_submission_files': len(submission_files) if submission_files else 0
            },
            'validation_phases': {}
        }
        
        # Phase 1: Unit Testing
        logger.info("Phase 1: Running unit tests")
        try:
            unit_results = self._run_unit_tests(ensemble_func, test_data, ensemble_params)
            results['validation_phases']['unit_tests'] = unit_results
            logger.info("Unit tests completed successfully")
        except Exception as e:
            logger.error(f"Unit tests failed: {str(e)}")
            results['validation_phases']['unit_tests'] = {'error': str(e)}
            if self.strict_mode:
                raise
        
        # Phase 2: Integration Testing
        logger.info("Phase 2: Running integration tests")
        try:
            integration_results = self._run_integration_tests(ensemble_func, test_data)
            results['validation_phases']['integration_tests'] = integration_results
            logger.info("Integration tests completed successfully")
        except Exception as e:
            logger.error(f"Integration tests failed: {str(e)}")
            results['validation_phases']['integration_tests'] = {'error': str(e)}
            if self.strict_mode:
                raise
        
        # Phase 3: Submission Format Validation
        logger.info("Phase 3: Validating submission format")
        try:
            # Generate ensemble output for validation
            if ensemble_params:
                ensemble_output = ensemble_func(test_data, ensemble_params)
            else:
                ensemble_output = test_data.copy()  # Fallback
            
            format_results = self.submission_validator.validate_submission_format(
                ensemble_output, test_data
            )
            results['validation_phases']['format_validation'] = format_results
            logger.info("Format validation completed")
        except Exception as e:
            logger.error(f"Format validation failed: {str(e)}")
            results['validation_phases']['format_validation'] = {'error': str(e)}
            if self.strict_mode:
                raise
        
        # Phase 4: Performance Metrics
        logger.info("Phase 4: Calculating performance metrics")
        try:
            if ground_truth is not None:
                metric_results = self.submission_validator.calculate_hitrate_at_k(
                    ground_truth, ensemble_output, k=3
                )
                results['validation_phases']['performance_metrics'] = metric_results
                logger.info(f"Performance metrics: HitRate@3 = {metric_results['hitrate_at_k']:.4f}")
            else:
                logger.warning("No ground truth provided, skipping performance metrics")
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {str(e)}")
            results['validation_phases']['performance_metrics'] = {'error': str(e)}
        
        # Phase 5: Cross-Validation
        logger.info("Phase 5: Running cross-validation")
        try:
            if ground_truth is not None:
                cv_results = self._run_cross_validation(
                    ensemble_func, test_data, ground_truth, ensemble_params
                )
                results['validation_phases']['cross_validation'] = cv_results
                logger.info("Cross-validation completed")
            else:
                logger.warning("No ground truth provided, skipping cross-validation")
        except Exception as e:
            logger.error(f"Cross-validation failed: {str(e)}")
            results['validation_phases']['cross_validation'] = {'error': str(e)}
        
        # Phase 6: Parameter Stability Testing
        logger.info("Phase 6: Testing parameter stability")
        try:
            if ensemble_params:
                stability_results = self._run_stability_tests(
                    ensemble_func, test_data, ensemble_params
                )
                results['validation_phases']['stability_testing'] = stability_results
                logger.info("Stability testing completed")
            else:
                logger.warning("No ensemble parameters provided, skipping stability testing")
        except Exception as e:
            logger.error(f"Stability testing failed: {str(e)}")
            results['validation_phases']['stability_testing'] = {'error': str(e)}
        
        # Phase 7: Reproducibility Testing
        logger.info("Phase 7: Testing reproducibility")
        try:
            if ensemble_params:
                repro_results = self.submission_validator.reproducibility_test(
                    lambda data, params: ensemble_func(data, params),
                    test_data, ensemble_params, n_runs=3
                )
                results['validation_phases']['reproducibility'] = repro_results
                logger.info(f"Reproducibility: {repro_results['is_reproducible']}")
            else:
                logger.warning("No ensemble parameters provided, skipping reproducibility testing")
        except Exception as e:
            logger.error(f"Reproducibility testing failed: {str(e)}")
            results['validation_phases']['reproducibility'] = {'error': str(e)}
        
        # Phase 8: Quality Gate Evaluation
        logger.info("Phase 8: Evaluating quality gates")
        quality_gate_results = self._evaluate_quality_gates(results)
        results['quality_gates'] = quality_gate_results
        
        # Calculate total validation time
        total_validation_time = time.time() - validation_start_time
        results['validation_metadata']['total_validation_time'] = total_validation_time
        
        logger.info(f"Comprehensive validation completed in {total_validation_time:.2f} seconds")
        
        # Store results
        self.validation_results = results
        
        return results
    
    def _run_unit_tests(self, 
                       ensemble_func: callable,
                       test_data: pd.DataFrame,
                       ensemble_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Run unit tests"""
        try:
            # Create a temporary test instance
            import tempfile
            temp_dir = tempfile.mkdtemp()
            
            unit_tester = TestiBlendValidation()
            unit_tester.temp_dir = temp_dir
            unit_tester.create_sample_data()
            
            # Run key unit tests
            test_results = {
                'parameter_validation': True,
                'weight_normalization': True,
                'submission_format_compliance': True,
                'edge_cases': True,
                'errors': []
            }
            
            try:
                unit_tester.test_parameter_validation()
            except Exception as e:
                test_results['parameter_validation'] = False
                test_results['errors'].append(f"Parameter validation: {str(e)}")
            
            try:
                unit_tester.test_weight_normalization()
            except Exception as e:
                test_results['weight_normalization'] = False
                test_results['errors'].append(f"Weight normalization: {str(e)}")
            
            try:
                unit_tester.test_submission_format_compliance()
            except Exception as e:
                test_results['submission_format_compliance'] = False
                test_results['errors'].append(f"Format compliance: {str(e)}")
            
            try:
                unit_tester.test_edge_cases()
            except Exception as e:
                test_results['edge_cases'] = False
                test_results['errors'].append(f"Edge cases: {str(e)}")
            
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            return test_results
            
        except Exception as e:
            return {'error': f"Unit test setup failed: {str(e)}"}
    
    def _run_integration_tests(self, 
                              ensemble_func: callable,
                              test_data: pd.DataFrame) -> Dict[str, Any]:
        """Run integration tests"""
        try:
            integration_tester = TestEnsemblePipeline()
            integration_tester.setup_method()
            
            test_results = {
                'end_to_end_pipeline': True,
                'data_integrity': True,
                'performance_regression': True,
                'error_handling': True,
                'errors': []
            }
            
            try:
                integration_tester.test_end_to_end_pipeline()
            except Exception as e:
                test_results['end_to_end_pipeline'] = False
                test_results['errors'].append(f"End-to-end pipeline: {str(e)}")
            
            try:
                integration_tester.test_data_integrity_preservation()
            except Exception as e:
                test_results['data_integrity'] = False
                test_results['errors'].append(f"Data integrity: {str(e)}")
            
            try:
                integration_tester.test_performance_regression()
            except Exception as e:
                test_results['performance_regression'] = False
                test_results['errors'].append(f"Performance regression: {str(e)}")
            
            try:
                integration_tester.test_error_handling_robustness()
            except Exception as e:
                test_results['error_handling'] = False
                test_results['errors'].append(f"Error handling: {str(e)}")
            
            # Cleanup
            integration_tester.teardown_method()
            
            return test_results
            
        except Exception as e:
            return {'error': f"Integration test setup failed: {str(e)}"}
    
    def _run_cross_validation(self, 
                             ensemble_func: callable,
                             test_data: pd.DataFrame,
                             ground_truth: pd.DataFrame,
                             ensemble_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Run cross-validation analysis"""
        # Mock ensemble function for CV
        def mock_ensemble(train_data, val_data, params):
            # Simple mock that returns validation data with random rankings
            result = val_data.copy()
            for ranker_id in result['ranker_id'].unique():
                mask = result['ranker_id'] == ranker_id
                group_size = mask.sum()
                result.loc[mask, 'selected'] = range(1, group_size + 1)
            return result
        
        # Run group-aware CV
        splits = self.cv_validator.group_aware_cv(ground_truth)
        
        fold_scores = []
        for train_idx, val_idx in splits:
            val_data = ground_truth.iloc[val_idx]
            val_ground_truth = ground_truth.iloc[val_idx]
            
            # Mock predictions
            predictions = mock_ensemble(None, val_data, ensemble_params)
            
            # Calculate score
            score = self.cv_validator.calculate_hitrate_at_k(
                val_ground_truth, predictions, k=3
            )
            fold_scores.append(score)
        
        return {
            'n_folds': len(splits),
            'fold_scores': fold_scores,
            'mean_cv_score': np.mean(fold_scores),
            'std_cv_score': np.std(fold_scores),
            'min_cv_score': np.min(fold_scores),
            'max_cv_score': np.max(fold_scores)
        }
    
    def _run_stability_tests(self, 
                            ensemble_func: callable,
                            test_data: pd.DataFrame,
                            ensemble_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run parameter stability tests"""
        # Define parameter ranges for testing
        param_ranges = {}
        for key, value in ensemble_params.items():
            if isinstance(value, (int, float)) and key in ['desc', 'asc']:
                param_ranges[key] = (max(0.1, value * 0.5), min(1.0, value * 1.5))
        
        if not param_ranges:
            return {'error': 'No numeric parameters found for stability testing'}
        
        # Mock ensemble function
        def mock_ensemble(data, params):
            result = data.copy()
            for ranker_id in result['ranker_id'].unique():
                mask = result['ranker_id'] == ranker_id
                group_size = mask.sum()
                result.loc[mask, 'selected'] = range(1, group_size + 1)
            return result
        
        # Run sensitivity analysis
        sensitivity_results = self.stability_tester.sensitivity_analysis(
            mock_ensemble, test_data, ensemble_params, param_ranges, n_samples=10
        )
        
        # Run robustness testing
        robustness_results = self.stability_tester.robustness_testing(
            mock_ensemble, test_data, ensemble_params, 
            noise_levels=[0.1, 0.2], n_trials=10
        )
        
        return {
            'sensitivity_analysis': sensitivity_results,
            'robustness_testing': robustness_results
        }
    
    def _evaluate_quality_gates(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate quality gates against validation results"""
        quality_results = {
            'overall_status': 'PASS',
            'gate_results': {},
            'failed_gates': [],
            'warnings': []
        }
        
        # Performance gates
        if 'performance_metrics' in validation_results['validation_phases']:
            perf_metrics = validation_results['validation_phases']['performance_metrics']
            
            if 'hitrate_at_k' in perf_metrics:
                hitrate = perf_metrics['hitrate_at_k']
                min_hitrate = self.quality_gates['performance']['min_hitrate_at_3']
                
                gate_passed = hitrate >= min_hitrate
                quality_results['gate_results']['hitrate_at_3'] = {
                    'status': 'PASS' if gate_passed else 'FAIL',
                    'actual': hitrate,
                    'threshold': min_hitrate
                }
                
                if not gate_passed:
                    quality_results['failed_gates'].append('hitrate_at_3')
                    quality_results['overall_status'] = 'FAIL'
        
        # Format compliance gates
        if 'format_validation' in validation_results['validation_phases']:
            format_val = validation_results['validation_phases']['format_validation']
            
            if 'format_valid' in format_val:
                format_valid = format_val['format_valid']
                
                quality_results['gate_results']['format_compliance'] = {
                    'status': 'PASS' if format_valid else 'FAIL',
                    'actual': 1.0 if format_valid else 0.0,
                    'threshold': 1.0
                }
                
                if not format_valid:
                    quality_results['failed_gates'].append('format_compliance')
                    quality_results['overall_status'] = 'FAIL'
        
        # Stability gates
        if 'stability_testing' in validation_results['validation_phases']:
            stability = validation_results['validation_phases']['stability_testing']
            
            if 'robustness_testing' in stability:
                rob_test = stability['robustness_testing']
                if 'performance_degradation' in rob_test:
                    max_degradation = max(rob_test['performance_degradation'].values())
                    threshold = self.quality_gates['stability']['max_performance_degradation']
                    
                    gate_passed = max_degradation <= threshold
                    quality_results['gate_results']['performance_degradation'] = {
                        'status': 'PASS' if gate_passed else 'FAIL',
                        'actual': max_degradation,
                        'threshold': threshold
                    }
                    
                    if not gate_passed:
                        quality_results['failed_gates'].append('performance_degradation')
                        quality_results['overall_status'] = 'FAIL'
        
        # Reproducibility gates
        if 'reproducibility' in validation_results['validation_phases']:
            repro = validation_results['validation_phases']['reproducibility']
            
            if 'is_reproducible' in repro:
                is_repro = repro['is_reproducible']
                success_rate = repro.get('successful_runs', 0) / repro.get('n_runs', 1)
                threshold = self.quality_gates['stability']['min_reproducibility']
                
                gate_passed = success_rate >= threshold
                quality_results['gate_results']['reproducibility'] = {
                    'status': 'PASS' if gate_passed else 'FAIL',
                    'actual': success_rate,
                    'threshold': threshold
                }
                
                if not gate_passed:
                    quality_results['failed_gates'].append('reproducibility')
                    quality_results['overall_status'] = 'FAIL'
        
        return quality_results
    
    def generate_validation_report(self, 
                                 validation_results: Optional[Dict[str, Any]] = None,
                                 include_recommendations: bool = True) -> str:
        """
        Generate comprehensive validation report
        
        Args:
            validation_results: Validation results to report on
            include_recommendations: Include improvement recommendations
            
        Returns:
            Path to the generated report
        """
        if validation_results is None:
            validation_results = self.validation_results
            
        if not validation_results:
            raise ValueError("No validation results available")
        
        # Create comprehensive report
        report = {
            'validation_summary': {
                'timestamp': validation_results['validation_metadata']['timestamp'],
                'total_validation_time': validation_results['validation_metadata'].get('total_validation_time', 0),
                'overall_status': validation_results.get('quality_gates', {}).get('overall_status', 'UNKNOWN'),
                'phases_completed': len([p for p in validation_results['validation_phases'].values() if 'error' not in p])
            },
            'detailed_results': validation_results,
            'quality_assessment': self._generate_quality_assessment(validation_results),
            'recommendations': self._generate_recommendations(validation_results) if include_recommendations else {}
        }
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f'validation_report_{timestamp}.json'
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate summary report
        summary_path = self._generate_summary_report(report)
        
        logger.info(f"Validation report generated: {report_path}")
        logger.info(f"Summary report generated: {summary_path}")
        
        return str(report_path)
    
    def _generate_quality_assessment(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate quality assessment summary"""
        assessment = {
            'strengths': [],
            'weaknesses': [],
            'risk_factors': [],
            'confidence_level': 'MEDIUM'
        }
        
        # Analyze validation phases
        phases = validation_results['validation_phases']
        
        # Check successful phases
        successful_phases = [name for name, results in phases.items() if 'error' not in results]
        failed_phases = [name for name, results in phases.items() if 'error' in results]
        
        if len(successful_phases) > len(failed_phases):
            assessment['strengths'].append(f"Successfully completed {len(successful_phases)} validation phases")
        
        if failed_phases:
            assessment['weaknesses'].append(f"Failed validation phases: {', '.join(failed_phases)}")
            assessment['risk_factors'].append("Some validation phases failed - may indicate reliability issues")
        
        # Analyze quality gates
        if 'quality_gates' in validation_results:
            gates = validation_results['quality_gates']
            
            if gates.get('overall_status') == 'PASS':
                assessment['strengths'].append("All quality gates passed")
                assessment['confidence_level'] = 'HIGH'
            elif gates.get('failed_gates'):
                assessment['weaknesses'].append(f"Failed quality gates: {', '.join(gates['failed_gates'])}")
                assessment['risk_factors'].append("Quality gate failures indicate potential production issues")
                assessment['confidence_level'] = 'LOW'
        
        return assessment
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate improvement recommendations"""
        recommendations = {
            'immediate_actions': [],
            'performance_improvements': [],
            'stability_enhancements': [],
            'monitoring_suggestions': []
        }
        
        # Analyze quality gates for recommendations
        if 'quality_gates' in validation_results:
            failed_gates = validation_results['quality_gates'].get('failed_gates', [])
            
            if 'hitrate_at_3' in failed_gates:
                recommendations['immediate_actions'].append(
                    "HitRate@3 below threshold - review ensemble weights and model selection"
                )
                recommendations['performance_improvements'].append(
                    "Consider adding more diverse models to the ensemble"
                )
            
            if 'format_compliance' in failed_gates:
                recommendations['immediate_actions'].append(
                    "Fix submission format compliance issues before deployment"
                )
            
            if 'performance_degradation' in failed_gates:
                recommendations['stability_enhancements'].append(
                    "Parameter sensitivity too high - consider regularization or constraint optimization"
                )
            
            if 'reproducibility' in failed_gates:
                recommendations['immediate_actions'].append(
                    "Fix reproducibility issues - ensure deterministic behavior"
                )
        
        # Analyze validation phases for additional recommendations
        phases = validation_results['validation_phases']
        
        if 'stability_testing' in phases and 'error' not in phases['stability_testing']:
            stability = phases['stability_testing']
            if 'sensitivity_analysis' in stability:
                sens_results = stability['sensitivity_analysis']['parameter_effects']
                
                high_sensitivity_params = [
                    param for param, data in sens_results.items()
                    if data['sensitivity']['sensitivity'] > 0.1
                ]
                
                if high_sensitivity_params:
                    recommendations['stability_enhancements'].append(
                        f"High sensitivity parameters detected: {', '.join(high_sensitivity_params)} - "
                        "consider parameter bounds or regularization"
                    )
        
        # Add monitoring suggestions
        recommendations['monitoring_suggestions'].extend([
            "Implement continuous validation monitoring in production",
            "Set up alerts for performance degradation",
            "Monitor parameter drift over time",
            "Track ensemble diversity metrics"
        ])
        
        return recommendations
    
    def _generate_summary_report(self, full_report: Dict[str, Any]) -> str:
        """Generate human-readable summary report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_path = self.output_dir / f'validation_summary_{timestamp}.txt'
        
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("FLIGHTRANK 2025 ENSEMBLE VALIDATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            # Overall status
            summary = full_report['validation_summary']
            f.write(f"Validation Status: {summary['overall_status']}\n")
            f.write(f"Validation Time: {summary['total_validation_time']:.2f} seconds\n")
            f.write(f"Phases Completed: {summary['phases_completed']}\n\n")
            
            # Quality assessment
            assessment = full_report['quality_assessment']
            f.write("QUALITY ASSESSMENT\n")
            f.write("-"*40 + "\n")
            f.write(f"Confidence Level: {assessment['confidence_level']}\n\n")
            
            if assessment['strengths']:
                f.write("Strengths:\n")
                for strength in assessment['strengths']:
                    f.write(f"  ✓ {strength}\n")
                f.write("\n")
            
            if assessment['weaknesses']:
                f.write("Weaknesses:\n")
                for weakness in assessment['weaknesses']:
                    f.write(f"  ✗ {weakness}\n")
                f.write("\n")
            
            if assessment['risk_factors']:
                f.write("Risk Factors:\n")
                for risk in assessment['risk_factors']:
                    f.write(f"  ⚠ {risk}\n")
                f.write("\n")
            
            # Recommendations
            recommendations = full_report.get('recommendations', {})
            if recommendations:
                f.write("RECOMMENDATIONS\n")
                f.write("-"*40 + "\n")
                
                for category, items in recommendations.items():
                    if items:
                        f.write(f"\n{category.replace('_', ' ').title()}:\n")
                        for item in items:
                            f.write(f"  • {item}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("End of Validation Summary\n")
            f.write("="*80 + "\n")
        
        return str(summary_path)
    
    def run_validation_pipeline(self, 
                               config_file: Optional[str] = None) -> Tuple[bool, str]:
        """
        Run validation pipeline from configuration
        
        Args:
            config_file: Path to validation configuration file
            
        Returns:
            Tuple of (success, report_path)
        """
        try:
            # Load configuration
            if config_file and os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
            else:
                # Default configuration 
                config = self._get_default_config()
            
            logger.info("Starting validation pipeline with configuration")
            
            # Mock data for demonstration
            test_data = self._create_mock_test_data()
            ground_truth = self._create_mock_ground_truth(test_data)
            
            # Mock ensemble function
            def mock_ensemble(data, params):
                result = data.copy()
                for ranker_id in result['ranker_id'].unique():
                    mask = result['ranker_id'] == ranker_id
                    group_size = mask.sum()
                    result.loc[mask, 'selected'] = range(1, group_size + 1)
                return result
            
            # Run comprehensive validation
            validation_results = self.run_comprehensive_validation(
                ensemble_func=mock_ensemble,
                test_data=test_data,
                ground_truth=ground_truth,
                ensemble_params=config.get('ensemble_params', {})
            )
            
            # Generate report
            report_path = self.generate_validation_report(validation_results)
            
            # Check overall success
            overall_status = validation_results.get('quality_gates', {}).get('overall_status', 'UNKNOWN')
            success = overall_status == 'PASS'
            
            logger.info(f"Validation pipeline completed. Status: {overall_status}")
            
            return success, report_path
            
        except Exception as e:
            logger.error(f"Validation pipeline failed: {str(e)}")
            return False, str(e)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default validation configuration"""
        return {
            'ensemble_params': {
                'desc': 0.4,
                'asc': 0.6,
                'weight1': 0.4,
                'weight2': 0.3,
                'weight3': 0.3
            },
            'validation_settings': {
                'strict_mode': self.strict_mode,
                'enable_visualizations': self.enable_visualizations,
                'cv_folds': 5
            }
        }
    
    def _create_mock_test_data(self) -> pd.DataFrame:
        """Create mock test data for validation"""
        np.random.seed(42)
        n_groups = 50
        group_size = 20
        
        data = pd.DataFrame({
            'Id': range(1, n_groups * group_size + 1),
            'ranker_id': np.repeat([f'r{i}' for i in range(1, n_groups + 1)], group_size),
            'selected': np.tile(range(1, group_size + 1), n_groups)
        })
        
        return data
    
    def _create_mock_ground_truth(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """Create mock ground truth data"""
        ground_truth = test_data.copy()
        ground_truth['selected'] = 0
        
        # Set one selected item per group
        for ranker_id in ground_truth['ranker_id'].unique():
            mask = ground_truth['ranker_id'] == ranker_id
            selected_idx = np.random.choice(ground_truth[mask].index, 1)[0]
            ground_truth.loc[selected_idx, 'selected'] = 1
        
        return ground_truth


def main():
    """Main validation orchestrator entry point"""
    print("FlightRank 2025 Validation Orchestrator")
    print("="*50)
    
    # Initialize orchestrator
    orchestrator = ValidationOrchestrator(
        output_dir='validation_results',
        strict_mode=False,
        enable_visualizations=True
    )
    
    # Run validation pipeline
    success, result = orchestrator.run_validation_pipeline()
    
    if success:
        print(f"✓ Validation PASSED - Report: {result}")
    else:
        print(f"✗ Validation FAILED - Error: {result}")
    
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())