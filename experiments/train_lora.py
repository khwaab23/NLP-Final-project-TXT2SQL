"""
Train model with LoRA
"""

import argparse
import yaml
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.utils.data_loader import SpiderDataset
from src.training.lora_trainer import LoRATrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Text-to-SQL model with LoRA")
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/phi-2",
        choices=["microsoft/phi-2", "meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1"],
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
        default="./results/lora",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of training samples (for testing)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*50)
    print(f"Training {args.model} with LoRA")
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
        tokenizer=tokenizer,
        max_length=2048
    )
    
    eval_dataset = SpiderDataset(
        args.data_path,
        split="dev",
        tokenizer=tokenizer,
        max_length=2048
    )
    
    if args.max_samples:
        train_dataset.examples = train_dataset.examples[:args.max_samples]
        eval_dataset.examples = eval_dataset.examples[:min(args.max_samples//10, len(eval_dataset.examples))]
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # Configure LoRA
    lora_config = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    
    # Configure training
    training_config = {
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size * 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": args.learning_rate,
        "fp16": True,
        "logging_steps": 10,
        "save_steps": 500,
        "eval_steps": 100,
        "gradient_checkpointing": True,
    }
    
    # Create trainer
    print("\nInitializing LoRA trainer...")
    trainer = LoRATrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        lora_config=lora_config,
        training_config=training_config,
        output_dir=args.output_dir
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    print("\nTraining completed!")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
