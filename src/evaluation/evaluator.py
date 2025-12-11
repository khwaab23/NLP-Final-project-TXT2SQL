"""
Model Evaluator
"""

import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import pandas as pd

from .metrics import EvaluationMetrics, BatchEvaluator


class ModelEvaluator:
    """Evaluate Text-to-SQL models"""
    
    def __init__(
        self,
        model,
        dataset,
        db_base_path: Optional[str] = None,
        batch_size: int = 1
    ):
        self.model = model
        self.dataset = dataset
        self.db_base_path = db_base_path
        self.batch_evaluator = BatchEvaluator(db_base_path)
        self.batch_size = batch_size
    
    def evaluate(
        self,
        num_samples: Optional[int] = None,
        metrics: List[str] = None,
        save_predictions: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model on dataset
        
        Args:
            num_samples: Number of samples to evaluate (None = all)
            metrics: List of metrics to calculate
            save_predictions: Whether to save predictions
        
        Returns:
            Evaluation results
        """
        if metrics is None:
            metrics = ['exact_match', 'execution_accuracy', 'component_match', 'valid_sql']
        
        predictions = []
        ground_truths = []
        db_ids = []
        inference_times = []
        
        # Determine number of samples
        total_samples = min(num_samples, len(self.dataset)) if num_samples else len(self.dataset)
        
        print(f"Evaluating on {total_samples} samples...")
        
        for i in tqdm(range(total_samples)):
            example = self.dataset[i]
            
            question = example['question']
            schema = example.get('schema', '')
            ground_truth = example['sql']
            db_id = example.get('db_id', '')
            
            # Generate prediction
            start_time = time.time()
            try:
                prediction = self.model.generate_sql(question, schema)
            except Exception as e:
                print(f"Error generating SQL for example {i}: {e}")
                prediction = ""
            
            inference_time = time.time() - start_time
            
            predictions.append(prediction)
            ground_truths.append(ground_truth)
            db_ids.append(db_id)
            inference_times.append(inference_time)
        
        # Calculate metrics
        results = self.batch_evaluator.evaluate_batch(
            predictions,
            ground_truths,
            db_ids,
            metrics
        )
        
        # Add timing information
        results['avg_inference_time'] = sum(inference_times) / len(inference_times)
        results['total_time'] = sum(inference_times)
        
        # Add model information
        if hasattr(self.model, 'get_stats'):
            results['model_stats'] = self.model.get_stats()
        
        # Save predictions if requested
        if save_predictions:
            self.save_predictions(predictions, ground_truths, results)
        
        return results
    
    def save_predictions(
        self,
        predictions: List[str],
        ground_truths: List[str],
        metrics: Dict[str, Any],
        output_file: str = "predictions.csv"
    ):
        """Save predictions and metrics to file"""
        data = {
            'prediction': predictions,
            'ground_truth': ground_truths,
        }
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
        # Save metrics
        metrics_file = output_file.replace('.csv', '_metrics.json')
        import json
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_file}")
    
    def compare_models(
        self,
        models: Dict[str, Any],
        num_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            models: Dictionary of {model_name: model}
            num_samples: Number of samples to evaluate
        
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            evaluator = ModelEvaluator(model, self.dataset, self.db_base_path)
            model_results = evaluator.evaluate(num_samples=num_samples)
            
            model_results['model'] = model_name
            results.append(model_results)
        
        df = pd.DataFrame(results)
        return df
    
    def error_analysis(
        self,
        predictions: List[str],
        ground_truths: List[str],
        questions: List[str],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze common errors
        
        Args:
            predictions: List of predictions
            ground_truths: List of ground truths
            questions: List of questions
            top_k: Number of top errors to return
        
        Returns:
            Error analysis results
        """
        errors = []
        
        for pred, gt, question in zip(predictions, ground_truths, questions):
            if not EvaluationMetrics.exact_match(pred, gt):
                error_type = self._classify_error(pred, gt)
                errors.append({
                    'question': question,
                    'predicted': pred,
                    'ground_truth': gt,
                    'error_type': error_type,
                    'component_match': EvaluationMetrics.component_match(pred, gt)
                })
        
        # Aggregate error types
        error_types = {}
        for error in errors:
            error_type = error['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_errors': len(errors),
            'error_types': error_types,
            'top_errors': errors[:top_k],
            'error_rate_by_type': {k: v/len(predictions) for k, v in error_types.items()}
        }
    
    def _classify_error(self, predicted: str, ground_truth: str) -> str:
        """Classify type of error"""
        if not predicted or len(predicted.strip()) == 0:
            return "empty_prediction"
        
        if not EvaluationMetrics.exact_match(predicted, ground_truth):
            pred_components = EvaluationMetrics.keyword_accuracy(predicted, ground_truth)
            
            if not pred_components.get('FROM', True):
                return "missing_from"
            elif not pred_components.get('WHERE', True):
                return "where_clause_mismatch"
            elif not pred_components.get('JOIN', True):
                return "join_mismatch"
            elif not pred_components.get('GROUP BY', True):
                return "groupby_mismatch"
            else:
                return "other_mismatch"
        
        return "unknown"
