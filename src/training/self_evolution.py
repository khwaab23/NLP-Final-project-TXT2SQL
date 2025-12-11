"""
Self-Evolution Training Loop for rStar-SQL

Implements iterative training where policy model and Process Preference Model
evolve together across multiple rounds.

Each round:
1. Generate CoT data using current policy + MCTS
2. Train policy on new CoT data
3. Collect trajectory pairs for PPM
4. Train PPM on preference pairs
5. Repeat
"""

import os
import json
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import wandb

from .rstar_sql import MCTS_SQL, ProcessPreferenceModel, SQLState
from .cot_synthesis import CoTSynthesizer, CoTExample, CoTDataGenerator
from ..utils.sql_executor import SQLExecutor


@dataclass
class EvolutionConfig:
    """Configuration for self-evolution training"""
    
    num_rounds: int = 5
    cot_examples_per_round: int = 1000
    policy_train_epochs: int = 3
    ppm_train_epochs: int = 2
    mcts_simulations: int = 100
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    max_length: int = 512
    save_every_round: bool = True
    eval_every_round: bool = True
    use_wandb: bool = False


class TrajectoryPairCollector:
    """
    Collects trajectory pairs for PPM training
    
    Generates pairs of (better_trajectory, worse_trajectory) based on:
    - Final SQL execution results
    - Intermediate step quality
    - Reward signals
    """
    
    def __init__(self, sql_executor: SQLExecutor):
        self.sql_executor = sql_executor
    
    def collect_pairs(
        self,
        cot_examples: List[CoTExample],
        num_pairs: int = 500
    ) -> List[Tuple[List[SQLState], List[SQLState]]]:
        """
        Collect preference pairs from CoT examples
        
        Returns:
            List of (preferred_traj, dispreferred_traj) pairs
        """
        pairs = []
        
        # Group examples by question
        question_groups = self._group_by_question(cot_examples)
        
        # Create pairs within each question group
        for question, examples in question_groups.items():
            if len(examples) < 2:
                continue
            
            # Sort by reward
            examples.sort(key=lambda e: e.reward, reverse=True)
            
            # Create pairs: high-reward vs low-reward
            num_pairs_for_question = min(3, len(examples) - 1)
            
            for i in range(num_pairs_for_question):
                better = examples[i]
                worse = examples[-(i+1)]
                
                # Convert to trajectory
                better_traj = self._example_to_trajectory(better)
                worse_traj = self._example_to_trajectory(worse)
                
                pairs.append((better_traj, worse_traj))
                
                if len(pairs) >= num_pairs:
                    break
            
            if len(pairs) >= num_pairs:
                break
        
        return pairs
    
    def _group_by_question(
        self,
        examples: List[CoTExample]
    ) -> Dict[str, List[CoTExample]]:
        """Group examples by question"""
        groups = {}
        
        for example in examples:
            q = example.question
            if q not in groups:
                groups[q] = []
            groups[q].append(example)
        
        return groups
    
    def _example_to_trajectory(self, example: CoTExample) -> List[SQLState]:
        """Convert CoTExample to trajectory of SQLStates"""
        trajectory = []
        
        for state_dict in example.trajectory:
            state = SQLState(
                partial_sql=state_dict['partial_sql'],
                schema=example.schema,
                question=example.question,
                db_id=example.db_id,
                step=state_dict['step'],
                components=state_dict['components'],
                is_terminal=state_dict['is_terminal'],
                reward=state_dict.get('reward', 0.0)
            )
            trajectory.append(state)
        
        return trajectory


class PolicyTrainer:
    """Trains the policy model on CoT data"""
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
    
    def train(
        self,
        cot_examples: List[CoTExample],
        config: EvolutionConfig
    ):
        """Train policy model on CoT examples"""
        print(f"Training policy model on {len(cot_examples)} CoT examples...")
        
        # Generate training data
        data_generator = CoTDataGenerator(cot_examples)
        train_data = data_generator.generate_instruction_data()
        
        # Prepare optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        # Training loop
        self.model.train()
        
        for epoch in range(config.policy_train_epochs):
            total_loss = 0
            num_batches = 0
            
            # Create batches
            for i in tqdm(range(0, len(train_data), config.batch_size)):
                batch = train_data[i:i + config.batch_size]
                
                # Prepare inputs
                texts = [
                    f"{item['instruction']}\n\n{item['response']}"
                    for item in batch
                ]
                
                inputs = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=config.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{config.policy_train_epochs}, Loss: {avg_loss:.4f}")
    
    def save(self, output_dir: str):
        """Save model checkpoint"""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Saved policy model to {output_dir}")


class PPMTrainer:
    """Trains the Process Preference Model"""
    
    def __init__(
        self,
        ppm: ProcessPreferenceModel,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.ppm = ppm
        self.device = device
    
    def train(
        self,
        trajectory_pairs: List[Tuple[List[SQLState], List[SQLState]]],
        config: EvolutionConfig
    ):
        """
        Train PPM on trajectory preference pairs
        
        Uses Bradley-Terry preference learning
        """
        print(f"Training PPM on {len(trajectory_pairs)} trajectory pairs...")
        
        optimizer = torch.optim.AdamW(
            self.ppm.model.parameters(),
            lr=config.learning_rate
        )
        
        self.ppm.model.train()
        
        for epoch in range(config.ppm_train_epochs):
            total_loss = 0
            num_batches = 0
            
            for i in tqdm(range(0, len(trajectory_pairs), config.batch_size)):
                batch = trajectory_pairs[i:i + config.batch_size]
                
                batch_loss = 0
                
                for better_traj, worse_traj in batch:
                    # Get preference score
                    pref_score = self.ppm.compare_trajectories(better_traj, worse_traj)
                    
                    # Loss: -log(preference for better)
                    loss = -torch.log(torch.tensor(pref_score + 1e-8))
                    batch_loss += loss
                
                # Normalize by batch size
                batch_loss = batch_loss / len(batch)
                
                # Backward pass
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                total_loss += batch_loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{config.ppm_train_epochs}, PPM Loss: {avg_loss:.4f}")
    
    def save(self, output_dir: str):
        """Save PPM checkpoint"""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.ppm.model.state_dict(), f"{output_dir}/ppm_model.pt")
        print(f"Saved PPM to {output_dir}")


class SelfEvolutionTrainer:
    """
    Main self-evolution training loop
    
    Coordinates:
    - CoT synthesis
    - Policy training
    - PPM training
    - Evaluation
    """
    
    def __init__(
        self,
        policy_model,
        tokenizer,
        sql_executor: SQLExecutor,
        config: EvolutionConfig,
        output_dir: str = "checkpoints/rstar"
    ):
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.sql_executor = sql_executor
        self.config = config
        self.output_dir = output_dir
        
        # Initialize components
        self.ppm = ProcessPreferenceModel(policy_model, tokenizer)
        self.mcts = MCTS_SQL(
            policy_model=policy_model,
            ppm=self.ppm,
            sql_executor=sql_executor,
            num_simulations=config.mcts_simulations
        )
        self.cot_synthesizer = CoTSynthesizer(
            mcts=self.mcts,
            ppm=self.ppm,
            sql_executor=sql_executor
        )
        
        # Trainers
        self.policy_trainer = PolicyTrainer(policy_model, tokenizer)
        self.ppm_trainer = PPMTrainer(self.ppm)
        
        # Collectors
        self.pair_collector = TrajectoryPairCollector(sql_executor)
        
        # Logging
        if config.use_wandb:
            wandb.init(project="rstar-sql", config=vars(config))
    
    def train(
        self,
        train_questions: List[Dict[str, Any]],
        db_paths: Dict[str, str],
        eval_questions: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Run self-evolution training loop
        
        Args:
            train_questions: Training questions for CoT synthesis
            db_paths: Mapping of db_id to database paths
            eval_questions: Optional evaluation questions
        """
        print("=" * 80)
        print("Starting rStar-SQL Self-Evolution Training")
        print("=" * 80)
        
        for round_num in range(1, self.config.num_rounds + 1):
            print(f"\n{'='*80}")
            print(f"Round {round_num}/{self.config.num_rounds}")
            print(f"{'='*80}\n")
            
            # Step 1: Generate CoT data with current policy
            print(f"[Round {round_num}] Step 1: Synthesizing CoT data...")
            cot_examples = self.cot_synthesizer.synthesize_dataset(
                questions=train_questions,
                db_paths=db_paths,
                target_size=self.config.cot_examples_per_round
            )
            
            # Save CoT data
            cot_path = f"{self.output_dir}/round_{round_num}_cot_data.json"
            self.cot_synthesizer.save_dataset(cot_examples, cot_path)
            
            # Step 2: Train policy on CoT data
            print(f"[Round {round_num}] Step 2: Training policy model...")
            self.policy_trainer.train(cot_examples, self.config)
            
            # Step 3: Collect trajectory pairs for PPM
            print(f"[Round {round_num}] Step 3: Collecting trajectory pairs...")
            trajectory_pairs = self.pair_collector.collect_pairs(
                cot_examples,
                num_pairs=500
            )
            
            # Step 4: Train PPM
            print(f"[Round {round_num}] Step 4: Training Process Preference Model...")
            self.ppm_trainer.train(trajectory_pairs, self.config)
            
            # Step 5: Evaluate
            if self.config.eval_every_round and eval_questions:
                print(f"[Round {round_num}] Step 5: Evaluating...")
                metrics = self._evaluate(eval_questions, db_paths)
                print(f"Evaluation metrics: {metrics}")
                
                if self.config.use_wandb:
                    wandb.log({f"round_{round_num}": metrics})
            
            # Step 6: Save checkpoints
            if self.config.save_every_round:
                round_dir = f"{self.output_dir}/round_{round_num}"
                self.policy_trainer.save(round_dir)
                self.ppm_trainer.save(round_dir)
                print(f"[Round {round_num}] Saved checkpoints to {round_dir}")
        
        print("\n" + "="*80)
        print("Self-Evolution Training Complete!")
        print("="*80)
    
    def _evaluate(
        self,
        eval_questions: List[Dict[str, Any]],
        db_paths: Dict[str, str]
    ) -> Dict[str, float]:
        """Evaluate current model"""
        correct = 0
        total = len(eval_questions[:50])  # Sample 50 for faster eval
        
        for question_data in eval_questions[:50]:
            sql, _ = self.mcts.search(
                question=question_data['question'],
                schema=question_data['schema'],
                db_id=question_data['db_id'],
                db_path=db_paths.get(question_data['db_id'], "")
            )
            
            # Check if SQL is correct (simple check)
            success, _, _ = self.sql_executor.execute_query(
                sql,
                db_paths.get(question_data['db_id'], "")
            )
            
            if success:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'execution_accuracy': accuracy,
            'num_evaluated': total
        }


# Export
__all__ = [
    'EvolutionConfig',
    'SelfEvolutionTrainer',
    'PolicyTrainer',
    'PPMTrainer',
    'TrajectoryPairCollector'
]
