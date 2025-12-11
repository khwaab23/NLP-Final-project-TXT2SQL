"""
GRPO (Group Relative Policy Optimization) Trainer for Text-to-SQL
"""

import torch
from transformers import Trainer, TrainingArguments
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from typing import Dict, Any, Optional, List
import numpy as np

from ..evaluation.metrics import EvaluationMetrics
from ..utils.sql_executor import SQLExecutor


class GRPOTrainer:
    """
    Trainer for GRPO (Group Relative Policy Optimization)
    
    GRPO is a reinforcement learning approach that uses relative rewards
    within groups of examples to improve SQL generation.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset=None,
        grpo_config: Dict[str, Any] = None,
        output_dir: str = "./results/grpo"
    ):
        self.base_model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        
        # Default GRPO configuration
        default_grpo_config = {
            "learning_rate": 1e-5,
            "batch_size": 4,
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
            "group_size": 4,  # Number of samples in a group for relative comparison
        }
        self.grpo_config = {**default_grpo_config, **(grpo_config or {})}
        
        self.model = None
        self.ppo_trainer = None
    
    def prepare_model(self):
        """Prepare model for GRPO training"""
        print("Preparing model for GRPO training...")
        
        # Create model with value head for PPO
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(self.base_model)
        
        print("Model prepared with value head for reinforcement learning")
        return self.model
    
    def setup_trainer(self):
        """Setup PPO Trainer for GRPO"""
        if self.model is None:
            self.prepare_model()
        
        # Create PPO config
        ppo_config = PPOConfig(
            model_name=self.output_dir,
            learning_rate=self.grpo_config["learning_rate"],
            batch_size=self.grpo_config["batch_size"],
            mini_batch_size=self.grpo_config["mini_batch_size"],
            gradient_accumulation_steps=self.grpo_config["gradient_accumulation_steps"],
            ppo_epochs=self.grpo_config["ppo_epochs"],
            max_grad_norm=self.grpo_config["max_grad_norm"],
            vf_coef=self.grpo_config["vf_coef"],
            cliprange=self.grpo_config["cliprange"],
            cliprange_value=self.grpo_config["cliprange_value"],
            gamma=self.grpo_config["gamma"],
            lam=self.grpo_config["lam"],
            kl_penalty=self.grpo_config["kl_penalty"],
            target_kl=self.grpo_config["target_kl"],
        )
        
        # Create PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=None,  # Will use the same model as reference
            tokenizer=self.tokenizer,
        )
        
        return self.ppo_trainer
    
    def compute_reward(
        self,
        generated_sql: str,
        ground_truth_sql: str,
        db_path: Optional[str] = None
    ) -> float:
        """
        Compute reward for generated SQL
        
        Combines multiple metrics:
        - Execution accuracy: 1.0
        - Exact match: 0.8
        - Component match: 0.5 * score
        - Valid SQL: 0.3
        """
        reward = 0.0
        
        # Check execution accuracy
        if db_path:
            try:
                exec_acc = EvaluationMetrics.execution_accuracy(
                    generated_sql, ground_truth_sql, db_path
                )
                if exec_acc:
                    reward += 1.0
            except:
                pass
        
        # Check exact match
        if EvaluationMetrics.exact_match(generated_sql, ground_truth_sql):
            reward += 0.8
        
        # Component match
        component_score = EvaluationMetrics.component_match(generated_sql, ground_truth_sql)
        reward += 0.5 * component_score
        
        # Valid SQL
        if SQLExecutor.is_valid_sql(generated_sql):
            reward += 0.3
        
        return reward
    
    def compute_group_relative_rewards(
        self,
        group_sqls: List[str],
        ground_truths: List[str],
        db_paths: Optional[List[str]] = None
    ) -> List[float]:
        """
        Compute relative rewards within a group
        
        This is the key difference in GRPO: rewards are normalized
        within groups to reduce variance.
        """
        # Compute raw rewards
        raw_rewards = []
        for i, (sql, gt) in enumerate(zip(group_sqls, ground_truths)):
            db_path = db_paths[i] if db_paths else None
            reward = self.compute_reward(sql, gt, db_path)
            raw_rewards.append(reward)
        
        # Normalize within group (mean 0, std 1)
        rewards = np.array(raw_rewards)
        if rewards.std() > 0:
            normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            normalized_rewards = rewards - rewards.mean()
        
        return normalized_rewards.tolist()
    
    def train_step(self, batch_examples: List[Dict[str, Any]]) -> Dict[str, float]:
        """Single training step for a batch"""
        group_size = self.grpo_config["group_size"]
        
        # Process in groups
        num_groups = len(batch_examples) // group_size
        total_reward = 0.0
        
        for group_idx in range(num_groups):
            start_idx = group_idx * group_size
            end_idx = start_idx + group_size
            group = batch_examples[start_idx:end_idx]
            
            # Prepare inputs
            queries = []
            ground_truths = []
            db_paths = []
            
            for example in group:
                prompt = f"Question: {example['question']}\nSchema: {example['schema']}\nSQL: "
                queries.append(prompt)
                ground_truths.append(example['sql'])
                db_paths.append(example.get('db_path'))
            
            # Generate SQL queries
            query_tensors = [self.tokenizer.encode(q, return_tensors="pt").to(self.model.device) for q in queries]
            response_tensors = self.ppo_trainer.generate(query_tensors, **self.generation_kwargs)
            
            # Decode responses
            generated_sqls = [self.tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
            
            # Compute group relative rewards
            rewards = self.compute_group_relative_rewards(generated_sqls, ground_truths, db_paths)
            
            # Update model
            reward_tensors = [torch.tensor(r).to(self.model.device) for r in rewards]
            stats = self.ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
            
            total_reward += sum(rewards)
        
        return {
            "avg_reward": total_reward / len(batch_examples),
            "ppo_stats": stats
        }
    
    def train(self, num_epochs: int = 3):
        """Train the model using GRPO"""
        if self.ppo_trainer is None:
            self.setup_trainer()
        
        # Generation kwargs
        self.generation_kwargs = {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
        }
        
        print("Starting GRPO training...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Shuffle dataset
            import random
            examples = list(self.train_dataset)
            random.shuffle(examples)
            
            # Train in batches
            batch_size = self.grpo_config["batch_size"]
            num_batches = len(examples) // batch_size
            
            epoch_rewards = []
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch = examples[start_idx:end_idx]
                
                stats = self.train_step(batch)
                epoch_rewards.append(stats["avg_reward"])
                
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}/{num_batches}, Avg Reward: {stats['avg_reward']:.4f}")
            
            avg_epoch_reward = sum(epoch_rewards) / len(epoch_rewards)
            print(f"Epoch {epoch + 1} completed. Average Reward: {avg_epoch_reward:.4f}")
            
            # Save checkpoint
            self.save_model(f"{self.output_dir}/epoch_{epoch+1}")
        
        print("GRPO training completed!")
    
    def save_model(self, output_dir: Optional[str] = None):
        """Save trained model"""
        save_dir = output_dir or self.output_dir
        
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        print(f"Model saved to {save_dir}")
