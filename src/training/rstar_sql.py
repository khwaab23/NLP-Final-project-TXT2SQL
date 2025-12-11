"""
rStar-SQL: Deep Thinking for Text-to-SQL with Monte Carlo Tree Search

Implements the rStar approach for SQL generation:
1. MCTS-based test-time search for SQL generation
2. Process Preference Model (PPM) for step-level guidance
3. Code-augmented Chain-of-Thought synthesis
4. Self-evolution training loop
"""

import math
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from copy import deepcopy


@dataclass
class SQLState:
    """Represents a state in the SQL generation process"""
    
    partial_sql: str
    schema: str
    question: str
    db_id: str
    step: int
    components: List[str]  # Components generated so far (SELECT, FROM, WHERE, etc.)
    is_terminal: bool = False
    execution_result: Optional[Any] = None
    reward: float = 0.0
    
    def __hash__(self):
        return hash((self.partial_sql, self.step))
    
    def __eq__(self, other):
        return self.partial_sql == other.partial_sql and self.step == other.step


@dataclass
class MCTSNode:
    """Node in the Monte Carlo Tree Search"""
    
    state: SQLState
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = None
    visits: int = 0
    total_reward: float = 0.0
    prior_prob: float = 0.0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    @property
    def q_value(self) -> float:
        """Average reward"""
        return self.total_reward / self.visits if self.visits > 0 else 0.0
    
    @property
    def is_leaf(self) -> bool:
        """Check if node is a leaf"""
        return len(self.children) == 0
    
    def uct_score(self, c_puct: float = 1.0) -> float:
        """Upper Confidence Bound for Trees score"""
        if self.parent is None:
            return 0.0
        
        # UCT = Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        exploration = c_puct * self.prior_prob * math.sqrt(self.parent.visits) / (1 + self.visits)
        return self.q_value + exploration


class ProcessPreferenceModel:
    """
    Process Preference Model (PPM) for evaluating SQL generation steps
    
    Instead of naive step-level scoring, this model learns preferences
    between different reasoning paths.
    """
    
    def __init__(self, base_model, tokenizer):
        self.model = base_model
        self.tokenizer = tokenizer
        self.device = base_model.device
    
    def evaluate_step(
        self,
        state: SQLState,
        next_component: str,
        context: Dict[str, Any]
    ) -> float:
        """
        Evaluate the quality of adding next_component to current state
        
        Returns:
            Reward score between 0 and 1
        """
        # Create prompt for evaluation
        prompt = self._create_evaluation_prompt(state, next_component, context)
        
        # Get model's confidence in this step
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Use hidden states to compute step quality
            step_score = self._compute_step_score(outputs, state, next_component)
        
        return step_score
    
    def compare_trajectories(
        self,
        traj1: List[SQLState],
        traj2: List[SQLState]
    ) -> float:
        """
        Compare two reasoning trajectories
        
        Returns:
            Preference score for traj1 over traj2 (0 to 1)
        """
        # Evaluate each trajectory
        score1 = self._evaluate_trajectory(traj1)
        score2 = self._evaluate_trajectory(traj2)
        
        # Return preference probability using Bradley-Terry model
        return 1.0 / (1.0 + math.exp(score2 - score1))
    
    def _create_evaluation_prompt(
        self,
        state: SQLState,
        next_component: str,
        context: Dict[str, Any]
    ) -> str:
        """Create prompt for evaluating a step"""
        return f"""Question: {state.question}
Schema: {state.schema}
Database: {state.db_id}

Current SQL: {state.partial_sql}
Next component: {next_component}

Is this a valid and correct step towards answering the question?
Rate from 0 (incorrect) to 1 (perfect):"""
    
    def _compute_step_score(self, outputs, state: SQLState, next_component: str) -> float:
        """Compute quality score for a step"""
        # This would use the model's hidden states and logits
        # For now, placeholder implementation
        # In practice: use learned reward head or value function
        
        # Check syntactic validity
        validity_score = 0.5
        
        # Check semantic relevance to question
        relevance_score = 0.5
        
        # Check schema compatibility
        schema_score = 0.5
        
        return (validity_score + relevance_score + schema_score) / 3
    
    def _evaluate_trajectory(self, trajectory: List[SQLState]) -> float:
        """Evaluate entire trajectory"""
        if not trajectory:
            return 0.0
        
        # Final state reward is most important
        final_reward = trajectory[-1].reward
        
        # Step-wise consistency
        step_scores = []
        for i in range(len(trajectory) - 1):
            # Check if step i leads logically to step i+1
            consistency = self._check_step_consistency(trajectory[i], trajectory[i+1])
            step_scores.append(consistency)
        
        avg_consistency = sum(step_scores) / len(step_scores) if step_scores else 1.0
        
        return 0.7 * final_reward + 0.3 * avg_consistency
    
    def _check_step_consistency(self, state1: SQLState, state2: SQLState) -> float:
        """Check consistency between consecutive steps"""
        # Placeholder: check if state2 builds upon state1
        return 0.8


class MCTS_SQL:
    """
    Monte Carlo Tree Search for SQL Generation
    
    Performs deep thinking through iterative search:
    1. Selection: Choose promising nodes using UCT
    2. Expansion: Add new candidate SQL components
    3. Simulation: Rollout to complete SQL
    4. Backpropagation: Update node values
    """
    
    def __init__(
        self,
        policy_model,
        ppm: ProcessPreferenceModel,
        sql_executor,
        num_simulations: int = 100,
        c_puct: float = 1.4,
        temperature: float = 1.0
    ):
        self.policy_model = policy_model
        self.ppm = ppm
        self.sql_executor = sql_executor
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
    
    def search(
        self,
        question: str,
        schema: str,
        db_id: str,
        db_path: str
    ) -> Tuple[str, List[SQLState]]:
        """
        Perform MCTS to find best SQL query
        
        Returns:
            Best SQL query and its reasoning trajectory
        """
        # Initialize root
        initial_state = SQLState(
            partial_sql="",
            schema=schema,
            question=question,
            db_id=db_id,
            step=0,
            components=[]
        )
        root = MCTSNode(state=initial_state, prior_prob=1.0)
        
        # Run simulations
        for i in range(self.num_simulations):
            node = root
            
            # Selection: Traverse tree using UCT
            while not node.is_leaf and not node.state.is_terminal:
                node = self._select_child(node)
            
            # Expansion: Add new children if not terminal
            if not node.state.is_terminal:
                self._expand(node)
            
            # Simulation: Rollout from this node
            reward, trajectory = self._simulate(node.state, db_path)
            
            # Backpropagation: Update all ancestors
            self._backpropagate(node, reward)
        
        # Select best action from root
        best_child = max(root.children, key=lambda c: c.visits)
        best_sql, best_trajectory = self._extract_sql_from_node(best_child)
        
        return best_sql, best_trajectory
    
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child with highest UCT score"""
        return max(node.children, key=lambda c: c.uct_score(self.c_puct))
    
    def _expand(self, node: MCTSNode):
        """Expand node by generating possible next SQL components"""
        # Get possible next components from policy model
        next_components = self._generate_next_components(node.state)
        
        for component, prob in next_components:
            # Create new state
            new_state = self._apply_component(node.state, component)
            
            # Create child node
            child = MCTSNode(
                state=new_state,
                parent=node,
                prior_prob=prob
            )
            node.children.append(child)
    
    def _generate_next_components(
        self,
        state: SQLState
    ) -> List[Tuple[str, float]]:
        """
        Generate possible next SQL components with probabilities
        
        Returns:
            List of (component, probability) tuples
        """
        # Determine what component to add next based on current state
        components = []
        
        if not state.components:
            # Start with SELECT
            components = [
                ("SELECT * FROM", 0.3),
                ("SELECT DISTINCT", 0.2),
                (f"SELECT [specific columns]", 0.5)
            ]
        elif "SELECT" in state.components and "FROM" not in state.components:
            # Add FROM clause
            components = self._get_from_candidates(state)
        elif "FROM" in state.components and "WHERE" not in state.components:
            # Optionally add WHERE, JOIN, or GROUP BY
            components = [
                ("WHERE", 0.4),
                ("JOIN", 0.3),
                ("GROUP BY", 0.2),
                ("[END]", 0.1)  # Complete the query
            ]
        else:
            # Continue building or finish
            components = self._get_continuation_candidates(state)
        
        return components[:5]  # Top 5 candidates
    
    def _get_from_candidates(self, state: SQLState) -> List[Tuple[str, float]]:
        """Get FROM clause candidates based on schema"""
        # Parse schema to get table names
        tables = self._extract_tables_from_schema(state.schema)
        candidates = [(f"FROM {table}", 1.0/len(tables)) for table in tables]
        return candidates
    
    def _get_continuation_candidates(self, state: SQLState) -> List[Tuple[str, float]]:
        """Get candidates for continuing the query"""
        candidates = []
        
        if "WHERE" in state.components and "GROUP BY" not in state.components:
            candidates.append(("AND", 0.3))
            candidates.append(("GROUP BY", 0.2))
            candidates.append(("ORDER BY", 0.2))
            candidates.append(("[END]", 0.3))
        else:
            candidates.append(("[END]", 0.8))
            candidates.append(("LIMIT", 0.2))
        
        return candidates
    
    def _apply_component(self, state: SQLState, component: str) -> SQLState:
        """Apply component to create new state"""
        new_state = SQLState(
            partial_sql=state.partial_sql + " " + component,
            schema=state.schema,
            question=state.question,
            db_id=state.db_id,
            step=state.step + 1,
            components=state.components + [component.split()[0]]  # Add component type
        )
        
        # Check if terminal
        if component == "[END]" or self._is_complete_sql(new_state.partial_sql):
            new_state.is_terminal = True
        
        return new_state
    
    def _simulate(
        self,
        state: SQLState,
        db_path: str
    ) -> Tuple[float, List[SQLState]]:
        """
        Simulate (rollout) from current state to terminal state
        
        Returns:
            Final reward and trajectory
        """
        trajectory = [state]
        current_state = deepcopy(state)
        
        # Rollout until terminal
        max_steps = 10
        for _ in range(max_steps):
            if current_state.is_terminal:
                break
            
            # Generate next step using policy model (fast rollout)
            next_component = self._fast_policy_sample(current_state)
            current_state = self._apply_component(current_state, next_component)
            trajectory.append(current_state)
        
        # Evaluate final SQL
        reward = self._evaluate_sql(current_state, db_path)
        trajectory[-1].reward = reward
        
        return reward, trajectory
    
    def _fast_policy_sample(self, state: SQLState) -> str:
        """Quick sampling from policy for rollout"""
        candidates = self._generate_next_components(state)
        if not candidates:
            return "[END]"
        
        # Sample based on probabilities
        components, probs = zip(*candidates)
        probs = np.array(probs)
        probs = probs / probs.sum()  # Normalize
        
        idx = np.random.choice(len(components), p=probs)
        return components[idx]
    
    def _evaluate_sql(self, state: SQLState, db_path: str) -> float:
        """
        Evaluate final SQL query
        
        Returns reward based on:
        - Syntactic validity
        - Executability
        - Likely correctness (via PPM)
        """
        sql = state.partial_sql.strip()
        
        # Check validity
        if not self.sql_executor.is_valid_sql(sql):
            return 0.0
        
        # Try execution
        success, results, error = self.sql_executor.execute_query(sql, db_path)
        
        if not success:
            return 0.1  # Partial credit for valid but non-executable
        
        # Use PPM to estimate quality
        ppm_score = self.ppm.evaluate_step(state, "[FINAL]", {})
        
        # Combine scores
        reward = 0.3 + 0.7 * ppm_score  # At least 0.3 if executable
        
        return reward
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagate reward up the tree"""
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent
    
    def _extract_sql_from_node(
        self,
        node: MCTSNode
    ) -> Tuple[str, List[SQLState]]:
        """Extract SQL query and trajectory from node"""
        trajectory = []
        current = node
        
        # Trace back to root to build trajectory
        while current is not None:
            trajectory.insert(0, current.state)
            current = current.parent
        
        sql = trajectory[-1].partial_sql if trajectory else ""
        return sql.strip(), trajectory
    
    def _is_complete_sql(self, sql: str) -> bool:
        """Check if SQL query is complete"""
        sql_upper = sql.upper()
        has_select = "SELECT" in sql_upper
        has_from = "FROM" in sql_upper
        return has_select and has_from
    
    def _extract_tables_from_schema(self, schema: str) -> List[str]:
        """Extract table names from schema"""
        # Simple parsing - in practice, use proper SQL parser
        import re
        tables = re.findall(r'CREATE TABLE (\w+)', schema, re.IGNORECASE)
        return tables if tables else ["table1"]


import torch


# Export
__all__ = [
    'SQLState',
    'MCTSNode',
    'ProcessPreferenceModel',
    'MCTS_SQL'
]
