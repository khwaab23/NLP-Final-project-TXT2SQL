"""
DoRA Trainer for Text-to-SQL
"""

import torch
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Dict, Any, Optional


class DoRATrainer:
    """Trainer for DoRA (Weight-Decomposed Low-Rank Adaptation)"""
    
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset=None,
        dora_config: Dict[str, Any] = None,
        training_config: Dict[str, Any] = None,
        output_dir: str = "./results/dora"
    ):
        self.base_model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        
        # Default DoRA configuration (similar to LoRA but with use_dora=True)
        default_dora_config = {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "use_dora": True,  # Key difference from LoRA
        }
        self.dora_config = {**default_dora_config, **(dora_config or {})}
        
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
        
        self.model = None
        self.trainer = None
    
    def prepare_model(self):
        """Prepare model for DoRA training"""
        print("Preparing model for DoRA training...")
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.base_model)
        
        # Create DoRA config (LoRA config with use_dora=True)
        peft_config = LoraConfig(
            r=self.dora_config["r"],
            lora_alpha=self.dora_config["lora_alpha"],
            target_modules=self.dora_config["target_modules"],
            lora_dropout=self.dora_config["lora_dropout"],
            bias=self.dora_config["bias"],
            task_type=self.dora_config["task_type"],
            use_dora=True,  # Enable DoRA
        )
        
        # Apply DoRA
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
        print("DoRA enabled: Weight decomposition into magnitude and direction")
        
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
        
        print("Starting DoRA training...")
        self.trainer.train()
        
        # Save the final model
        self.save_model()
        
        return self.trainer
    
    def save_model(self, output_dir: Optional[str] = None):
        """Save DoRA adapter"""
        save_dir = output_dir or self.output_dir
        
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        print(f"DoRA adapter saved to {save_dir}")
    
    def load_adapter(self, adapter_path: str):
        """Load trained DoRA adapter"""
        from peft import PeftModel
        
        self.model = PeftModel.from_pretrained(
            self.base_model,
            adapter_path,
            is_trainable=False
        )
        
        print(f"DoRA adapter loaded from {adapter_path}")
        return self.model
