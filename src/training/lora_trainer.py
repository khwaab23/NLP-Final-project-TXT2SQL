"""
LoRA Trainer for Text-to-SQL
"""

import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Dict, Any, Optional
import os


class LoRATrainer:
    """Trainer for LoRA fine-tuning"""
    
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset=None,
        lora_config: Dict[str, Any] = None,
        training_config: Dict[str, Any] = None,
        output_dir: str = "./results/lora"
    ):
        self.base_model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        
        # Default LoRA configuration
        default_lora_config = {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        self.lora_config = {**default_lora_config, **(lora_config or {})}
        
        # Default training configuration
        default_training_config = {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "fp16": True,
            "logging_steps": 10,
            "save_steps": 500,
            "eval_steps": 100,
        }
        self.training_config = {**default_training_config, **(training_config or {})}
        
        # Initialize model with LoRA
        self.model = None
        self.trainer = None
    
    def prepare_model(self):
        """Prepare model for LoRA training"""
        print("Preparing model for LoRA training...")
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.base_model)
        
        # Create LoRA config
        peft_config = LoraConfig(
            r=self.lora_config["r"],
            lora_alpha=self.lora_config["lora_alpha"],
            target_modules=self.lora_config["target_modules"],
            lora_dropout=self.lora_config["lora_dropout"],
            bias=self.lora_config["bias"],
            task_type=self.lora_config["task_type"],
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
        return self.model
    
    def setup_trainer(self):
        """Setup Hugging Face Trainer"""
        if self.model is None:
            self.prepare_model()
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.training_config["num_train_epochs"],
            per_device_train_batch_size=self.training_config["per_device_train_batch_size"],
            per_device_eval_batch_size=self.training_config.get("per_device_eval_batch_size", 8),
            gradient_accumulation_steps=self.training_config["gradient_accumulation_steps"],
            learning_rate=self.training_config["learning_rate"],
            weight_decay=self.training_config.get("weight_decay", 0.01),
            fp16=self.training_config.get("fp16", True),
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=self.training_config["logging_steps"],
            save_steps=self.training_config["save_steps"],
            eval_strategy="steps" if self.eval_dataset else "no",
            eval_steps=self.training_config.get("eval_steps", 100) if self.eval_dataset else None,
            save_total_limit=3,
            load_best_model_at_end=True if self.eval_dataset else False,
            report_to=["tensorboard"],
            remove_unused_columns=False,
            gradient_checkpointing=self.training_config.get("gradient_checkpointing", True),
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
        )
        
        return self.trainer
    
    def train(self):
        """Train the model"""
        if self.trainer is None:
            self.setup_trainer()
        
        print("Starting LoRA training...")
        self.trainer.train()
        
        # Save the final model
        self.save_model()
        
        return self.trainer
    
    def save_model(self, output_dir: Optional[str] = None):
        """Save LoRA adapter"""
        save_dir = output_dir or self.output_dir
        
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        print(f"LoRA adapter saved to {save_dir}")
    
    def load_adapter(self, adapter_path: str):
        """Load trained LoRA adapter"""
        from peft import PeftModel
        
        self.model = PeftModel.from_pretrained(
            self.base_model,
            adapter_path,
            is_trainable=False
        )
        
        print(f"LoRA adapter loaded from {adapter_path}")
        return self.model
