"""
Train model with GRPO (Group Relative Policy Optimization)
"""

import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.utils.data_loader import SpiderDataset
from src.training.grpo_trainer import GRPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Text-to-SQL model with GRPO")
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/spider",
        help="Path to Spider dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/grpo",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=4,
        help="Group size for relative rewards"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of training samples"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*50)
    print(f"Training {args.model} with GRPO")
    print("="*50)
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = SpiderDataset(
        args.data_path,
        split="train",
        tokenizer=None,  # GRPO handles tokenization differently
        max_length=2048
    )
    
    if args.max_samples:
        train_dataset.examples = train_dataset.examples[:args.max_samples]
    
    print(f"Train samples: {len(train_dataset)}")
    
    # Configure GRPO
    grpo_config = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "mini_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "ppo_epochs": 4,
        "max_grad_norm": 1.0,
        "vf_coef": 0.1,
        "cliprange": 0.2,
        "cliprange_value": 0.2,
        "gamma": 1.0,
        "lam": 0.95,
        "kl_penalty": "kl",
        "target_kl": 0.1,
        "group_size": args.group_size,
    }
    
    # Create trainer
    print("\nInitializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        grpo_config=grpo_config,
        output_dir=args.output_dir
    )
    
    # Train
    print("\nStarting training...")
    trainer.train(num_epochs=args.epochs)
    
    print("\nTraining completed!")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
