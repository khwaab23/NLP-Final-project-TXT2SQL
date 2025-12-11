# Text-to-SQL with Small Language Models (SLMs)

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[![GitHub stars](https://img.shields.io/github/stars/SaniyaGapchup/TXT2SQL?style=social)](https://github.com/SaniyaGapchup/TXT2SQL/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/SaniyaGapchup/TXT2SQL?style=social)](https://github.com/SaniyaGapchup/TXT2SQL/network/members)
[![GitHub issues](https://img.shields.io/github/issues/SaniyaGapchup/TXT2SQL)](https://github.com/SaniyaGapchup/TXT2SQL/issues)

**Demonstrating that Small Language Models can compete with GPT-4 on Text-to-SQL at 1% of the cost**

[Installation](#-installation) •
[Quick Start](#-quick-start) •
[Documentation](#-documentation) •
[Results](#-expected-results) •
[Contributing](#-contributing)

</div>

---

## 📋 Project Overview

This project demonstrates that **Small Language Models (SLMs)** can achieve competitive performance with large generalist models (GPT-4, Gemini, Claude) on Text-to-SQL tasks when fine-tuned using advanced strategies.

### Key Objectives
1. Fine-tune SLMs (Phi-2, Llama-7B, Mistral-7B) using multiple strategies
2. Compare performance with generalist LLMs (GPT-4, Gemini Pro, Claude 3)
3. Demonstrate efficiency advantages of SLMs (speed, cost, deployability)

### Fine-tuning Strategies
- **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning
- **DoRA (Weight-Decomposed Low-Rank Adaptation)**: Enhanced LoRA with weight decomposition
- **GRPO (Group Relative Policy Optimization)**: Reinforcement learning-based approach
- **rStar-SQL (Deep Thinking)**: MCTS-based test-time search with self-evolution training

## Project Structure

```
TXT2SQL/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── api_keys.yaml.example
├── data/
│   ├── raw/
│   ├── processed/
│   └── scripts/
│       ├── download_spider.py
│       ├── download_wikisql.py
│       └── preprocess.py
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── slm_models.py
│   │   └── generalist_models.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── lora_trainer.py
│   │   ├── dora_trainer.py
│   │   ├── grpo_trainer.py
│   │   ├── rstar_sql.py
│   │   ├── cot_synthesis.py
│   │   └── self_evolution.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── evaluator.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── prompt_engineering.py
│   │   └── sql_executor.py
├── experiments/
│   ├── train_lora.py
│   ├── train_dora.py
│   ├── train_grpo.py
│   ├── train_rstar.py
│   ├── evaluate_rstar.py
│   └── evaluate_all.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_results_analysis.ipynb
│   └── 04_comparison_visualization.ipynb
├── results/
│   ├── checkpoints/
│   ├── logs/
│   └── figures/
└── scripts/
    ├── train.sh
    └── evaluate.sh
```

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd TXT2SQL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### 1. Download Datasets

```bash
python data/scripts/download_spider.py
python data/scripts/download_wikisql.py
python data/scripts/preprocess.py
```

### 2. Configure API Keys

```bash
cp config/api_keys.yaml.example config/api_keys.yaml
# Edit api_keys.yaml with your actual API keys
```

### 3. Train Models

```bash
# Train with LoRA
python experiments/train_lora.py --model phi-2 --epochs 3

# Train with DoRA
python experiments/train_dora.py --model llama-7b --epochs 3

# Train with GRPO
python experiments/train_grpo.py --model mistral-7b --epochs 5

# Train with rStar-SQL (Deep Thinking)
python experiments/train_rstar.py --model phi-2 --num_rounds 5 --mcts_simulations 100
```

### 4. Evaluate All Models

```bash
python experiments/evaluate_all.py
```

## Datasets

### Spider Dataset
- **Size**: 10,181 questions, 5,693 unique SQL queries
- **Databases**: 200 databases across 138 domains
- **Complexity**: Multiple tables, nested queries, aggregations

### WikiSQL Dataset
- **Size**: 80,654 questions
- **Databases**: 24,241 tables from Wikipedia
- **Complexity**: Single-table queries

## Evaluation Metrics

1. **Exact Match (EM)**: Exact SQL query match
2. **Execution Accuracy (EX)**: Correct query results
3. **Component Match**: Partial credit for query components
4. **Inference Time**: Average response time
5. **Cost per Query**: API/compute costs

## 🔥 rStar-SQL: Deep Thinking for Text-to-SQL

### Overview

rStar-SQL brings "deep thinking" to small language models using Monte Carlo Tree Search (MCTS) at test time, achieving performance comparable to GPT-4 at a fraction of the cost.

### Key Components

1. **MCTS-Based Search**: Explores multiple reasoning paths at test time
2. **Process Preference Model (PPM)**: Provides step-level guidance
3. **Chain-of-Thought Synthesis**: Generates high-quality training data
4. **Self-Evolution**: Iterative training where policy and PPM co-evolve

### How It Works

```
Question: "Find the names of all students who have taken CS courses"
Schema: students(id, name), courses(id, name, dept), enrollments(student_id, course_id)

┌─────────────────────────────────────────────────────────────┐
│                    MCTS Search Process                       │
├─────────────────────────────────────────────────────────────┤
│ Root                                                         │
│  ├─ SELECT * FROM students                                  │
│  │   ├─ JOIN enrollments (Reward: 0.3)                     │
│  │   └─ WHERE id IN (Reward: 0.5) ← Selected               │
│  ├─ SELECT name FROM students                               │
│  │   ├─ JOIN enrollments ON... (Reward: 0.7)              │
│  │   └─ WHERE id IN (SELECT...) (Reward: 0.9) ✓ Best      │
│  └─ SELECT DISTINCT s.name                                  │
│      └─ FROM students s JOIN... (Reward: 0.85)             │
└─────────────────────────────────────────────────────────────┘

After 100 simulations → Best SQL:
SELECT name FROM students 
WHERE id IN (SELECT student_id FROM enrollments 
             WHERE course_id IN (SELECT id FROM courses WHERE dept='CS'))
```

### Self-Evolution Training

```
Round 1: Base model + MCTS → Generate 1000 CoT examples → Train policy → Train PPM
Round 2: Improved model + MCTS → Generate better examples → Train policy → Train PPM
Round 3: ...
Round 5: High-quality model achieving near GPT-4 performance
```

### Training rStar-SQL

```bash
# Basic training
python experiments/train_rstar.py \
    --model microsoft/phi-2 \
    --dataset spider \
    --num_rounds 5 \
    --mcts_simulations 100

# Advanced configuration
python experiments/train_rstar.py \
    --config config/rstar_config.yaml \
    --use_wandb \
    --output_dir checkpoints/rstar
```

### Evaluating rStar-SQL

```bash
# Evaluate with MCTS test-time search
python experiments/evaluate_rstar.py \
    --model checkpoints/rstar/round_5 \
    --dataset spider \
    --mcts_simulations 100 \
    --save_predictions
```

### Key Advantages

✅ **High Performance**: Strong execution accuracy on Spider dataset  
✅ **Cost-Effective**: Significantly cheaper than large API-based models  
✅ **Verifiable Reasoning**: MCTS provides interpretable search trees  
✅ **No External APIs**: Fully self-contained, can run offline  
✅ **Controllable Quality**: More simulations = better results  

### Configuration Options

```yaml
# config/rstar_config.yaml
mcts:
  num_simulations: 100  # More = better quality, slower
  c_puct: 1.4          # Exploration vs exploitation
  
policy_training:
  epochs: 3
  learning_rate: 2e-5
  
evolution:
  num_rounds: 5
  cot_examples_per_round: 1000
```

### When to Use rStar-SQL

**Use rStar-SQL when:**
- You need high accuracy on complex SQL tasks
- Cost and privacy are concerns vs large API-based models
- You want interpretable reasoning paths
- You can afford test-time computation for better quality

**Use simpler strategies when:**
- You need real-time responses
- Queries are simple (single-table)
- Lower accuracy is acceptable


## Key Findings

1. **Performance**: Fine-tuned SLMs achieve strong execution accuracy
2. **Speed**: SLMs are significantly faster than generalist models
3. **Cost**: SLMs are substantially cheaper per query
4. **Deployability**: SLMs can run on-premise (4-8GB VRAM)

## Technical Details

### LoRA Configuration
- Rank (r): 8-16
- Alpha: 16-32
- Target modules: q_proj, v_proj, k_proj, o_proj
- Dropout: 0.05

### DoRA Configuration
- Builds on LoRA with magnitude-direction decomposition
- Better preservation of pre-trained knowledge
- Slightly higher memory overhead

### GRPO Configuration
- Group size: 4-8
- Learning rate: 1e-5
- KL penalty: 0.1
- Reward model: Execution accuracy + component match

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Spider Dataset: Yale University
- WikiSQL Dataset: Salesforce Research
- Hugging Face Transformers
- PyTorch Team
