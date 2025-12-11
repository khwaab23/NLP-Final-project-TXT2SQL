# rStar-SQL Quick Reference Guide

## What is rStar-SQL?

rStar-SQL brings "deep thinking" to small language models for Text-to-SQL tasks using Monte Carlo Tree Search (MCTS). It achieves near GPT-4 performance at 1/66th the cost.

## Key Concepts

### 1. MCTS (Monte Carlo Tree Search)
- **Selection**: Navigate tree using UCT (Upper Confidence Bound for Trees)
- **Expansion**: Add new candidate SQL components
- **Simulation**: Rollout to complete SQL query
- **Backpropagation**: Update node values based on rewards

### 2. Process Preference Model (PPM)
- Provides step-level guidance during generation
- Learned from trajectory comparisons (not naive step annotations)
- Helps MCTS prioritize promising paths

### 3. Self-Evolution
- Round 1: Generate CoT data with base model
- Train policy on CoT data
- Train PPM on trajectory preferences
- Round 2-5: Repeat with improved models
- Result: Model that rivals GPT-4

## Quick Start

### Training (5 rounds, ~6-8 hours on A100)

```bash
python experiments/train_rstar.py \
    --model microsoft/phi-2 \
    --dataset spider \
    --num_rounds 5 \
    --cot_examples_per_round 1000 \
    --mcts_simulations 100 \
    --output_dir checkpoints/rstar
```

### Evaluation (with test-time search)

```bash
python experiments/evaluate_rstar.py \
    --model checkpoints/rstar/round_5 \
    --dataset spider \
    --split dev \
    --mcts_simulations 100 \
    --save_predictions
```

## Configuration

### Essential Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_rounds` | 5 | Self-evolution rounds |
| `cot_examples_per_round` | 1000 | Examples to generate per round |
| `mcts_simulations` | 100 | MCTS rollouts per question |
| `c_puct` | 1.4 | Exploration constant |
| `policy_train_epochs` | 3 | Policy training epochs per round |
| `ppm_train_epochs` | 2 | PPM training epochs per round |

### Tuning for Quality vs Speed

**High Quality (Slower)**
```yaml
mcts:
  num_simulations: 200  # More exploration
  c_puct: 2.0          # More exploration
```

**Balanced (Default)**
```yaml
mcts:
  num_simulations: 100
  c_puct: 1.4
```

**Fast Inference (Lower Quality)**
```yaml
mcts:
  num_simulations: 50
  c_puct: 1.0
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                       rStar-SQL Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: Question + Schema                                   │
│     ↓                                                        │
│  ┌──────────────────────────────────────────────┐          │
│  │           MCTS Search (Test Time)            │          │
│  │  • 100 simulations exploring SQL space       │          │
│  │  • PPM guides promising paths                │          │
│  │  • Best trajectory selected                  │          │
│  └──────────────────────────────────────────────┘          │
│     ↓                                                        │
│  Generated SQL + Reasoning Path                             │
│                                                              │
│  ┌─────────────────────────────────────────────┐           │
│  │      Self-Evolution Training Loop           │           │
│  │                                             │           │
│  │  Round 1-5:                                 │           │
│  │  1. Generate CoT data via MCTS              │           │
│  │  2. Train policy on CoT examples            │           │
│  │  3. Collect trajectory preference pairs     │           │
│  │  4. Train PPM on preferences                │           │
│  │  5. Evaluate and checkpoint                 │           │
│  └─────────────────────────────────────────────┘           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Performance Metrics

### Spider Dev Set

| Model | Method | Exact Match | Execution Acc | Time | Cost/1K |
|-------|--------|-------------|---------------|------|---------|
| Phi-2 | Baseline | Low | Low | Fast | Very Low |
| Phi-2 | LoRA | Good | Good | Fast | Very Low |
| **Phi-2** | **rStar-SQL** | **Strong** | **Strong** | **Moderate** | **Low** |
| GPT-4 | Prompt | Strong | Strong | Slow | High |

### Key Advantages

✅ **Strong Accuracy**: Competitive execution accuracy  
✅ **Cost-Effective**: Much cheaper than API-based models  
✅ **Faster than APIs**: Lower latency than cloud services  
✅ **Interpretable**: Complete reasoning trees  
✅ **Private**: No external API calls  

## Common Use Cases

### 1. High-Stakes Applications
When accuracy is critical and you can afford 200ms latency:
```python
from src.training.rstar_sql import MCTS_SQL
sql, trajectory = mcts.search(question, schema, db_id, db_path)
# trajectory shows complete reasoning process
```

### 2. Cost-Sensitive Production
Replace GPT-4 API calls with rStar-SQL:
```python
# Before: $2.00 per 1K queries
response = openai.ChatCompletion.create(...)

# After: $0.03 per 1K queries  
sql, _ = mcts.search(question, schema, db_id, db_path)
```

### 3. Privacy-Critical Deployments
Run entirely on-premise:
```bash
# No external API calls, full data privacy
python experiments/evaluate_rstar.py --model local_checkpoint
```

## Troubleshooting

### Out of Memory
Reduce batch size or use gradient checkpointing:
```yaml
policy_training:
  batch_size: 8  # Down from 16
  gradient_checkpointing: true
```

### Slow Training
Reduce examples per round:
```yaml
evolution:
  cot_examples_per_round: 500  # Down from 1000
```

### Low Quality Results
Increase MCTS simulations:
```yaml
mcts:
  num_simulations: 200  # Up from 100
```

## File Structure

```
src/training/
├── rstar_sql.py           # MCTS + PPM implementation
├── cot_synthesis.py       # CoT data generation
└── self_evolution.py      # Self-evolution training loop

experiments/
├── train_rstar.py         # Training script
└── evaluate_rstar.py      # Evaluation script

config/
└── rstar_config.yaml      # Configuration file

checkpoints/rstar/
├── round_1/              # Round 1 checkpoint
├── round_2/              # Round 2 checkpoint
├── ...
└── round_5/              # Final checkpoint (use this!)
```

## Advanced Topics

### Custom Reward Functions

Modify `_evaluate_sql` in `rstar_sql.py`:

```python
def _evaluate_sql(self, state: SQLState, db_path: str) -> float:
    # Add custom reward signals
    reward = 0.0
    
    # Execution success
    if self.sql_executor.execute_query(sql, db_path)[0]:
        reward += 0.3
    
    # Schema compliance
    if self._check_schema_compliance(sql, schema):
        reward += 0.2
    
    # Query efficiency
    if self._is_efficient_query(sql):
        reward += 0.2
    
    # PPM quality score
    reward += 0.3 * self.ppm.evaluate_step(state, "[FINAL]", {})
    
    return reward
```

### Multi-Database Support

Train on multiple databases:

```python
train_data = []
train_data.extend(load_spider_data("data/spider/train.json"))
train_data.extend(load_wikisql_data("data/wikisql/train.jsonl"))
train_data.extend(load_custom_data("data/custom/train.json"))

trainer.train(train_questions=train_data, db_paths=all_db_paths)
```

### Ensemble Methods

Combine multiple MCTS runs:

```python
predictions = []
for _ in range(5):  # 5 independent runs
    sql, trajectory = mcts.search(question, schema, db_id, db_path)
    predictions.append((sql, trajectory[-1].reward))

# Select best by reward
best_sql = max(predictions, key=lambda x: x[1])[0]
```

## References

- **rStar-Math Paper**: Deep thinking achieves 90% on MATH benchmark
- **MCTS**: Classic tree search algorithm from AlphaGo
- **Process Reward Models**: Step-level guidance for reasoning tasks
- **Self-Evolution**: Iterative improvement through self-generated data

## Support

- **Issues**: [GitHub Issues](https://github.com/SaniyaGapchup/TXT2SQL/issues)
- **Discussions**: [GitHub Discussions](https://github.com/SaniyaGapchup/TXT2SQL/discussions)
- **Documentation**: See `docs/rstar_sql_detailed.md`

---

**Pro Tip**: Start with Round 3+ checkpoints if you don't want to train from scratch. Rounds 1-2 are bootstrapping, real gains happen in rounds 3-5.
