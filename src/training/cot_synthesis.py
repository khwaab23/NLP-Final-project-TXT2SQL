"""
Code-Augmented Chain-of-Thought Synthesis for SQL Generation

Performs extensive MCTS rollouts to generate high-quality reasoning
trajectories for training the policy model.
"""

import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import random
from tqdm import tqdm

from .rstar_sql import MCTS_SQL, SQLState, ProcessPreferenceModel


@dataclass
class CoTExample:
    """A Chain-of-Thought training example"""
    
    question: str
    schema: str
    db_id: str
    reasoning_steps: List[str]
    final_sql: str
    trajectory: List[Dict[str, Any]]
    reward: float
    verified: bool
    
    def to_dict(self):
        return asdict(self)
    
    def to_training_text(self) -> str:
        """Convert to text format for training"""
        text = f"Question: {self.question}\n"
        text += f"Schema: {self.schema}\n\n"
        text += "Reasoning:\n"
        
        for i, step in enumerate(self.reasoning_steps, 1):
            text += f"{i}. {step}\n"
        
        text += f"\nFinal SQL: {self.final_sql}"
        return text


class CoTSynthesizer:
    """
    Synthesizes Chain-of-Thought examples using MCTS
    
    Process:
    1. Run extensive MCTS rollouts per question
    2. Select high-reward trajectories
    3. Convert to natural language reasoning steps
    4. Verify with execution
    5. Filter and rank examples
    """
    
    def __init__(
        self,
        mcts: MCTS_SQL,
        ppm: ProcessPreferenceModel,
        sql_executor,
        num_rollouts_per_question: int = 50,
        min_reward_threshold: float = 0.7,
        diversity_weight: float = 0.3
    ):
        self.mcts = mcts
        self.ppm = ppm
        self.sql_executor = sql_executor
        self.num_rollouts = num_rollouts_per_question
        self.min_reward = min_reward_threshold
        self.diversity_weight = diversity_weight
    
    def synthesize_dataset(
        self,
        questions: List[Dict[str, Any]],
        db_paths: Dict[str, str],
        target_size: int = 1000
    ) -> List[CoTExample]:
        """
        Generate synthetic CoT dataset from questions
        
        Args:
            questions: List of question dicts with 'question', 'schema', 'db_id'
            db_paths: Mapping of db_id to database file path
            target_size: Target number of examples to generate
        
        Returns:
            List of high-quality CoT examples
        """
        all_examples = []
        
        print(f"Synthesizing CoT dataset (target: {target_size} examples)...")
        
        for question_data in tqdm(questions):
            question = question_data['question']
            schema = question_data['schema']
            db_id = question_data['db_id']
            db_path = db_paths.get(db_id, "")
            
            # Generate multiple trajectories for this question
            examples = self._generate_question_examples(
                question, schema, db_id, db_path
            )
            
            all_examples.extend(examples)
            
            # Stop if we have enough
            if len(all_examples) >= target_size:
                break
        
        # Rank and select best examples
        selected = self._select_best_examples(all_examples, target_size)
        
        print(f"Generated {len(selected)} high-quality CoT examples")
        return selected
    
    def _generate_question_examples(
        self,
        question: str,
        schema: str,
        db_id: str,
        db_path: str
    ) -> List[CoTExample]:
        """Generate multiple CoT examples for a single question"""
        examples = []
        
        for _ in range(self.num_rollouts):
            # Run MCTS search
            sql, trajectory = self.mcts.search(
                question=question,
                schema=schema,
                db_id=db_id,
                db_path=db_path
            )
            
            # Get reward
            reward = trajectory[-1].reward if trajectory else 0.0
            
            # Skip low-quality trajectories
            if reward < self.min_reward:
                continue
            
            # Verify execution
            verified = self._verify_sql(sql, db_path)
            
            # Convert trajectory to reasoning steps
            reasoning_steps = self._trajectory_to_reasoning(trajectory)
            
            # Create example
            example = CoTExample(
                question=question,
                schema=schema,
                db_id=db_id,
                reasoning_steps=reasoning_steps,
                final_sql=sql,
                trajectory=[self._state_to_dict(s) for s in trajectory],
                reward=reward,
                verified=verified
            )
            
            examples.append(example)
        
        return examples
    
    def _trajectory_to_reasoning(self, trajectory: List[SQLState]) -> List[str]:
        """
        Convert SQL generation trajectory to natural language reasoning
        
        Makes the reasoning human-readable for training
        """
        reasoning_steps = []
        
        for i, state in enumerate(trajectory[1:], 1):  # Skip initial empty state
            step = self._state_to_reasoning_step(state, i)
            if step:
                reasoning_steps.append(step)
        
        return reasoning_steps
    
    def _state_to_reasoning_step(self, state: SQLState, step_num: int) -> str:
        """Convert a state to a reasoning step"""
        component = state.components[-1] if state.components else ""
        partial = state.partial_sql.strip()
        
        # Generate natural language explanation
        if component == "SELECT":
            return f"Start by selecting the relevant columns for the question."
        elif component == "FROM":
            table = partial.split("FROM")[-1].strip()
            return f"The data should come from the '{table}' table."
        elif component == "WHERE":
            return "Add conditions to filter the results based on the question requirements."
        elif component == "JOIN":
            return "Join additional tables to get related information."
        elif component == "GROUP":
            return "Group results to aggregate data as needed."
        elif component == "ORDER":
            return "Order the results appropriately."
        else:
            return f"Continue building the query: {partial[-50:]}"  # Last 50 chars
    
    def _verify_sql(self, sql: str, db_path: str) -> bool:
        """Verify SQL executes successfully"""
        if not sql or not db_path:
            return False
        
        success, _, _ = self.sql_executor.execute_query(sql, db_path)
        return success
    
    def _select_best_examples(
        self,
        examples: List[CoTExample],
        target_size: int
    ) -> List[CoTExample]:
        """
        Select best examples balancing quality and diversity
        
        Uses:
        - Reward score
        - Verification status
        - Diversity (different reasoning patterns)
        """
        # Filter verified examples
        verified = [e for e in examples if e.verified]
        
        if len(verified) <= target_size:
            return verified
        
        # Score each example
        scored = [(self._score_example(e, verified), e) for e in verified]
        scored.sort(reverse=True, key=lambda x: x[0])
        
        # Select top examples
        selected = [e for _, e in scored[:target_size]]
        
        return selected
    
    def _score_example(self, example: CoTExample, all_examples: List[CoTExample]) -> float:
        """
        Score example for selection
        
        Combines:
        - Quality (reward)
        - Diversity (uniqueness of reasoning)
        """
        quality_score = example.reward
        
        # Diversity: how different is this from others?
        diversity_score = self._compute_diversity(example, all_examples)
        
        total_score = (1 - self.diversity_weight) * quality_score + \
                     self.diversity_weight * diversity_score
        
        return total_score
    
    def _compute_diversity(
        self,
        example: CoTExample,
        all_examples: List[CoTExample]
    ) -> float:
        """Compute diversity score for an example"""
        # Measure how unique the reasoning pattern is
        
        # For simplicity: count unique SQL patterns
        sql_pattern = self._get_sql_pattern(example.final_sql)
        
        # Count similar patterns
        similar_count = sum(
            1 for e in all_examples
            if self._get_sql_pattern(e.final_sql) == sql_pattern
        )
        
        # Higher diversity if fewer similar examples
        diversity = 1.0 / (1.0 + similar_count)
        
        return diversity
    
    def _get_sql_pattern(self, sql: str) -> str:
        """Extract abstract pattern from SQL"""
        # Simplify SQL to pattern
        pattern = sql.upper()
        
        # Replace specific values with placeholders
        import re
        pattern = re.sub(r"'[^']*'", "'VALUE'", pattern)
        pattern = re.sub(r'\d+', 'NUM', pattern)
        pattern = re.sub(r'\b\w+\.\w+\b', 'TABLE.COL', pattern)
        
        return pattern
    
    def _state_to_dict(self, state: SQLState) -> Dict[str, Any]:
        """Convert SQLState to dict for serialization"""
        return {
            'partial_sql': state.partial_sql,
            'step': state.step,
            'components': state.components,
            'is_terminal': state.is_terminal,
            'reward': state.reward
        }
    
    def save_dataset(self, examples: List[CoTExample], output_path: str):
        """Save CoT dataset to file"""
        data = [e.to_dict() for e in examples]
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(examples)} CoT examples to {output_path}")
    
    def load_dataset(self, input_path: str) -> List[CoTExample]:
        """Load CoT dataset from file"""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        examples = [
            CoTExample(
                question=d['question'],
                schema=d['schema'],
                db_id=d['db_id'],
                reasoning_steps=d['reasoning_steps'],
                final_sql=d['final_sql'],
                trajectory=d['trajectory'],
                reward=d['reward'],
                verified=d['verified']
            )
            for d in data
        ]
        
        print(f"Loaded {len(examples)} CoT examples from {input_path}")
        return examples


class CoTDataGenerator:
    """
    Generate training data from CoT examples
    
    Converts CoT examples to various formats:
    - Instruction-following format
    - Step-by-step reasoning format
    - Q&A format
    """
    
    def __init__(self, examples: List[CoTExample]):
        self.examples = examples
    
    def generate_instruction_data(self) -> List[Dict[str, str]]:
        """Generate instruction-following dataset"""
        data = []
        
        for example in self.examples:
            # Format as instruction
            instruction = f"""Given the following database schema and question, generate a SQL query with step-by-step reasoning.

Schema: {example.schema}

Question: {example.question}"""
            
            # Format response with CoT
            response = "Let me solve this step by step:\n\n"
            for i, step in enumerate(example.reasoning_steps, 1):
                response += f"{i}. {step}\n"
            
            response += f"\nFinal SQL Query:\n{example.final_sql}"
            
            data.append({
                'instruction': instruction,
                'response': response,
                'db_id': example.db_id
            })
        
        return data
    
    def generate_step_by_step_data(self) -> List[Dict[str, Any]]:
        """Generate step-by-step reasoning dataset"""
        data = []
        
        for example in self.examples:
            steps = []
            
            for i, (reasoning, state) in enumerate(zip(
                example.reasoning_steps,
                example.trajectory[1:]  # Skip initial state
            )):
                steps.append({
                    'step_num': i + 1,
                    'reasoning': reasoning,
                    'partial_sql': state['partial_sql'],
                    'is_final': state['is_terminal']
                })
            
            data.append({
                'question': example.question,
                'schema': example.schema,
                'db_id': example.db_id,
                'steps': steps,
                'final_sql': example.final_sql
            })
        
        return data
    
    def generate_qa_pairs(self) -> List[Dict[str, str]]:
        """Generate simple Q&A pairs"""
        data = []
        
        for example in self.examples:
            data.append({
                'question': f"Schema: {example.schema}\n\nQuestion: {example.question}",
                'answer': example.final_sql,
                'db_id': example.db_id
            })
        
        return data


# Export
__all__ = [
    'CoTExample',
    'CoTSynthesizer',
    'CoTDataGenerator'
]
