"""
Evaluate rStar-SQL Model

Evaluates a trained rStar-SQL model using MCTS at test time.

Usage:
    python experiments/evaluate_rstar.py --model checkpoints/rstar/round_5 --dataset spider
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.training.rstar_sql import MCTS_SQL, ProcessPreferenceModel
from src.utils.sql_executor import SQLExecutor
from src.utils.data_loader import load_spider_data, load_wikisql_data
from src.evaluation.metrics import (
    exact_match,
    execution_accuracy,
    component_match,
    token_f1,
    valid_sql_rate
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate rStar-SQL model")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained rStar model checkpoint'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='spider',
        choices=['spider', 'wikisql'],
        help='Dataset to evaluate on'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='dev',
        choices=['train', 'dev', 'test'],
        help='Dataset split to use'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of samples to evaluate (None for all)'
    )
    parser.add_argument(
        '--mcts_simulations',
        type=int,
        default=100,
        help='Number of MCTS simulations per question'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='results/rstar_evaluation.json',
        help='Output file for results'
    )
    parser.add_argument(
        '--save_predictions',
        action='store_true',
        help='Save individual predictions'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    
    return parser.parse_args()


def load_test_data(dataset_name: str, split: str, data_dir: str = 'data'):
    """Load test dataset"""
    if dataset_name == 'spider':
        data_file = f"{data_dir}/spider/{split}.json"
        data = load_spider_data(data_file)
        db_dir = f"{data_dir}/spider/database"
    else:  # wikisql
        data_file = f"{data_dir}/wikisql/{split}.jsonl"
        data = load_wikisql_data(data_file)
        db_dir = f"{data_dir}/wikisql/database"
    
    # Create db_paths mapping
    db_paths = {}
    if os.path.exists(db_dir):
        for db_name in os.listdir(db_dir):
            db_path = os.path.join(db_dir, db_name)
            if os.path.isdir(db_path):
                db_file = os.path.join(db_path, f"{db_name}.sqlite")
                if os.path.exists(db_file):
                    db_paths[db_name] = db_file
    
    return data, db_paths


def evaluate_model(
    model,
    tokenizer,
    test_data: List[Dict[str, Any]],
    db_paths: Dict[str, str],
    args
) -> Dict[str, Any]:
    """Evaluate model with MCTS"""
    
    # Initialize components
    sql_executor = SQLExecutor()
    ppm = ProcessPreferenceModel(model, tokenizer)
    mcts = MCTS_SQL(
        policy_model=model,
        ppm=ppm,
        sql_executor=sql_executor,
        num_simulations=args.mcts_simulations
    )
    
    # Results storage
    predictions = []
    all_predictions = []
    all_references = []
    
    # Evaluate
    print(f"Evaluating on {len(test_data)} examples...")
    
    for i, example in enumerate(tqdm(test_data)):
        question = example['question']
        schema = example.get('schema', '')
        db_id = example.get('db_id', '')
        reference_sql = example.get('query', '')
        db_path = db_paths.get(db_id, '')
        
        # Generate SQL with MCTS
        try:
            predicted_sql, trajectory = mcts.search(
                question=question,
                schema=schema,
                db_id=db_id,
                db_path=db_path
            )
        except Exception as e:
            print(f"Error on example {i}: {e}")
            predicted_sql = ""
            trajectory = []
        
        # Store results
        all_predictions.append(predicted_sql)
        all_references.append(reference_sql)
        
        if args.save_predictions:
            predictions.append({
                'question': question,
                'db_id': db_id,
                'predicted_sql': predicted_sql,
                'reference_sql': reference_sql,
                'num_mcts_steps': len(trajectory),
                'final_reward': trajectory[-1].reward if trajectory else 0.0
            })
    
    # Compute metrics
    print("\nComputing metrics...")
    
    metrics = {
        'exact_match': exact_match(all_predictions, all_references),
        'execution_accuracy': execution_accuracy(
            all_predictions,
            all_references,
            db_paths,
            sql_executor
        ),
        'component_match': component_match(all_predictions, all_references),
        'token_f1': token_f1(all_predictions, all_references),
        'valid_sql_rate': valid_sql_rate(all_predictions, sql_executor)
    }
    
    results = {
        'model': args.model,
        'dataset': args.dataset,
        'split': args.split,
        'num_samples': len(test_data),
        'mcts_simulations': args.mcts_simulations,
        'metrics': metrics
    }
    
    if args.save_predictions:
        results['predictions'] = predictions
    
    return results


def main():
    args = parse_args()
    
    print("="*80)
    print("rStar-SQL Evaluation")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset} ({args.split} split)")
    print(f"MCTS simulations: {args.mcts_simulations}")
    print(f"Device: {args.device}")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device == 'cuda' else torch.float32,
        trust_remote_code=True,
        device_map=args.device
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded: {model.num_parameters() / 1e6:.2f}M parameters")
    
    # Load data
    print("\nLoading test data...")
    test_data, db_paths = load_test_data(args.dataset, args.split)
    
    if args.num_samples:
        test_data = test_data[:args.num_samples]
    
    print(f"Loaded {len(test_data)} test examples")
    
    # Evaluate
    results = evaluate_model(model, tokenizer, test_data, db_paths, args)
    
    # Print results
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    for metric, value in results['metrics'].items():
        print(f"{metric}: {value:.4f}")
    print("="*80)
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output_file}")


if __name__ == '__main__':
    main()
