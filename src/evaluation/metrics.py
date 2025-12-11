"""
Evaluation Metrics for Text-to-SQL
"""

from typing import List, Dict, Any, Tuple
import re
from collections import Counter
from ..utils.sql_executor import SQLExecutor


class EvaluationMetrics:
    """Collection of metrics for Text-to-SQL evaluation"""
    
    @staticmethod
    def exact_match(predicted: str, ground_truth: str) -> bool:
        """
        Exact match metric (case-insensitive, whitespace-normalized)
        """
        pred_normalized = SQLExecutor.normalize_sql(predicted)
        gt_normalized = SQLExecutor.normalize_sql(ground_truth)
        return pred_normalized == gt_normalized
    
    @staticmethod
    def execution_accuracy(
        predicted: str,
        ground_truth: str,
        db_path: str,
        timeout: int = 10
    ) -> bool:
        """
        Execution accuracy - check if both queries return same results
        """
        executor = SQLExecutor(db_path, timeout)
        
        # Execute both queries
        success_pred, results_pred, _ = executor.execute_query(predicted)
        success_gt, results_gt, _ = executor.execute_query(ground_truth)
        
        if not (success_pred and success_gt):
            return False
        
        return SQLExecutor.compare_results(results_pred, results_gt)
    
    @staticmethod
    def component_match(predicted: str, ground_truth: str) -> float:
        """
        Component-wise matching (partial credit)
        Returns score between 0 and 1
        """
        pred_components = SQLExecutor.extract_sql_components(predicted)
        gt_components = SQLExecutor.extract_sql_components(ground_truth)
        
        total = len(gt_components)
        matches = sum(1 for k, v in gt_components.items() if pred_components.get(k) == v)
        
        return matches / total if total > 0 else 0.0
    
    @staticmethod
    def token_f1(predicted: str, ground_truth: str) -> float:
        """
        Token-level F1 score
        """
        def tokenize(sql: str) -> List[str]:
            # Simple tokenization
            sql = re.sub(r'[^a-zA-Z0-9_*]', ' ', sql.upper())
            return sql.split()
        
        pred_tokens = Counter(tokenize(predicted))
        gt_tokens = Counter(tokenize(ground_truth))
        
        # Calculate overlap
        common = pred_tokens & gt_tokens
        num_common = sum(common.values())
        
        if num_common == 0:
            return 0.0
        
        precision = num_common / sum(pred_tokens.values()) if sum(pred_tokens.values()) > 0 else 0
        recall = num_common / sum(gt_tokens.values()) if sum(gt_tokens.values()) > 0 else 0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    @staticmethod
    def valid_sql_rate(predicted_queries: List[str]) -> float:
        """
        Percentage of syntactically valid SQL queries
        """
        valid_count = sum(1 for sql in predicted_queries if SQLExecutor.is_valid_sql(sql))
        return valid_count / len(predicted_queries) if predicted_queries else 0.0
    
    @staticmethod
    def keyword_accuracy(predicted: str, ground_truth: str) -> Dict[str, bool]:
        """
        Check presence of specific SQL keywords
        """
        keywords = ['SELECT', 'FROM', 'WHERE', 'JOIN', 'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT']
        
        pred_upper = predicted.upper()
        gt_upper = ground_truth.upper()
        
        accuracy = {}
        for keyword in keywords:
            gt_has = keyword in gt_upper
            pred_has = keyword in pred_upper
            accuracy[keyword] = (pred_has == gt_has)
        
        return accuracy
    
    @staticmethod
    def calculate_all_metrics(
        predicted: str,
        ground_truth: str,
        db_path: str = None
    ) -> Dict[str, Any]:
        """
        Calculate all metrics at once
        """
        metrics = {
            'exact_match': EvaluationMetrics.exact_match(predicted, ground_truth),
            'component_match': EvaluationMetrics.component_match(predicted, ground_truth),
            'token_f1': EvaluationMetrics.token_f1(predicted, ground_truth),
            'valid_sql': SQLExecutor.is_valid_sql(predicted),
        }
        
        if db_path:
            metrics['execution_accuracy'] = EvaluationMetrics.execution_accuracy(
                predicted, ground_truth, db_path
            )
        
        return metrics


class BatchEvaluator:
    """Evaluate model predictions in batch"""
    
    def __init__(self, db_base_path: str = None):
        self.db_base_path = db_base_path
    
    def evaluate_batch(
        self,
        predictions: List[str],
        ground_truths: List[str],
        db_ids: List[str] = None,
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a batch of predictions
        
        Args:
            predictions: List of predicted SQL queries
            ground_truths: List of ground truth SQL queries
            db_ids: List of database IDs (for execution accuracy)
            metrics: List of metrics to calculate (default: all)
        
        Returns:
            Dictionary of metric scores
        """
        if metrics is None:
            metrics = ['exact_match', 'component_match', 'token_f1', 'valid_sql']
        
        results = {metric: [] for metric in metrics}
        
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            db_path = None
            if db_ids and self.db_base_path:
                db_path = f"{self.db_base_path}/{db_ids[i]}/{db_ids[i]}.sqlite"
            
            example_metrics = EvaluationMetrics.calculate_all_metrics(pred, gt, db_path)
            
            for metric in metrics:
                if metric in example_metrics:
                    results[metric].append(float(example_metrics[metric]))
        
        # Calculate averages
        avg_results = {}
        for metric, values in results.items():
            if values:
                avg_results[f"{metric}_avg"] = sum(values) / len(values)
                avg_results[f"{metric}_count"] = len(values)
        
        return avg_results
    
    def detailed_evaluation(
        self,
        predictions: List[Dict[str, Any]],
        include_errors: bool = True
    ) -> Dict[str, Any]:
        """
        Detailed evaluation with per-example results
        
        Args:
            predictions: List of dicts with 'predicted', 'ground_truth', 'question', etc.
            include_errors: Whether to include error analysis
        
        Returns:
            Detailed evaluation results
        """
        results = {
            'total': len(predictions),
            'metrics': {},
            'by_difficulty': {},
            'errors': [] if include_errors else None
        }
        
        all_metrics = []
        
        for i, example in enumerate(predictions):
            pred = example.get('predicted', '')
            gt = example.get('ground_truth', '')
            db_path = example.get('db_path')
            
            metrics = EvaluationMetrics.calculate_all_metrics(pred, gt, db_path)
            all_metrics.append(metrics)
            
            # Track errors
            if include_errors and not metrics.get('exact_match', False):
                results['errors'].append({
                    'index': i,
                    'question': example.get('question'),
                    'predicted': pred,
                    'ground_truth': gt,
                    'metrics': metrics
                })
        
        # Aggregate metrics
        for metric_name in all_metrics[0].keys():
            values = [m[metric_name] for m in all_metrics]
            results['metrics'][metric_name] = {
                'mean': sum(values) / len(values) if values else 0,
                'total_correct': sum(1 for v in values if v) if all(isinstance(v, bool) for v in values) else None
            }
        
        return results
