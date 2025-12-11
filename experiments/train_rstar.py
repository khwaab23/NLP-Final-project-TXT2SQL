"""
Train rStar-SQL: Deep Thinking for Text-to-SQL

This script implements the rStar approach using MCTS-based test-time search
and self-evolution training.

Usage:
    python experiments/train_rstar.py --config config/rstar_config.yaml
"""

import os
import sys
import argparse
import yaml
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.training.self_evolution import SelfEvolutionTrainer, EvolutionConfig
from src.utils.sql_executor import SQLExecutor
from src.utils.data_loader import load_spider_data, load_wikisql_data


def parse_args():
    parser = argparse.ArgumentParser(description="Train rStar-SQL model")
    parser.add_argument(
        '--config',
        type=str,
        default='config/rstar_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='microsoft/phi-2',
        help='Base model to use'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='checkpoints/rstar',
        help='Output directory for checkpoints'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='spider',
        choices=['spider', 'wikisql'],
        help='Dataset to use'
    )
    parser.add_argument(
        '--num_rounds',
        type=int,
        default=5,
        help='Number of self-evolution rounds'
    )
    parser.add_argument(
        '--cot_examples_per_round',
        type=int,
        default=1000,
        help='Number of CoT examples to generate per round'
    )
    parser.add_argument(
        '--mcts_simulations',
        type=int,
        default=100,
        help='Number of MCTS simulations per question'
    )
    parser.add_argument(
        '--use_wandb',
        action='store_true',
        help='Use Weights & Biases for logging'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def prepare_data(dataset_name: str, data_dir: str = 'data'):
    """Load and prepare dataset"""
    print(f"Loading {dataset_name} dataset...")
    
    if dataset_name == 'spider':
        train_data = load_spider_data(f"{data_dir}/spider/train_spider.json")
        dev_data = load_spider_data(f"{data_dir}/spider/dev.json")
        db_dir = f"{data_dir}/spider/database"
    else:  # wikisql
        train_data = load_wikisql_data(f"{data_dir}/wikisql/train.jsonl")
        dev_data = load_wikisql_data(f"{data_dir}/wikisql/dev.jsonl")
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
    
    print(f"Loaded {len(train_data)} training examples")
    print(f"Loaded {len(dev_data)} dev examples")
    print(f"Found {len(db_paths)} databases")
    
    return train_data, dev_data, db_paths


def main():
    args = parse_args()
    
    # Load config
    config_dict = load_config(args.config)
    
    # Create evolution config
    evolution_config = EvolutionConfig(
        num_rounds=args.num_rounds,
        cot_examples_per_round=args.cot_examples_per_round,
        mcts_simulations=args.mcts_simulations,
        policy_train_epochs=config_dict.get('policy_train_epochs', 3),
        ppm_train_epochs=config_dict.get('ppm_train_epochs', 2),
        batch_size=config_dict.get('batch_size', 16),
        learning_rate=config_dict.get('learning_rate', 2e-5),
        use_wandb=args.use_wandb
    )
    
    print("="*80)
    print("rStar-SQL Training Configuration")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Num rounds: {evolution_config.num_rounds}")
    print(f"CoT examples per round: {evolution_config.cot_examples_per_round}")
    print(f"MCTS simulations: {evolution_config.mcts_simulations}")
    print(f"Device: {args.device}")
    print("="*80)
    
    # Load base model
    print("\nLoading base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.device == 'cuda' else torch.float32,
        trust_remote_code=True,
        device_map=args.device
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    print(f"Loaded model: {args.model_name}")
    print(f"Model parameters: {model.num_parameters() / 1e6:.2f}M")
    
    # Prepare data
    train_data, dev_data, db_paths = prepare_data(args.dataset)
    
    # Initialize SQL executor
    sql_executor = SQLExecutor()
    
    # Create trainer
    print("\nInitializing rStar-SQL trainer...")
    trainer = SelfEvolutionTrainer(
        policy_model=model,
        tokenizer=tokenizer,
        sql_executor=sql_executor,
        config=evolution_config,
        output_dir=args.output_dir
    )
    
    # Start training
    print("\nStarting self-evolution training...\n")
    trainer.train(
        train_questions=train_data,
        db_paths=db_paths,
        eval_questions=dev_data
    )
    
    print("\n" + "="*80)
    print("Training complete! Checkpoints saved to:", args.output_dir)
    print("="*80)


if __name__ == '__main__':
    main()
