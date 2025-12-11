# Quick Start Guide

Get started with Text-to-SQL using Small Language Models in under 10 minutes!

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (8GB+ VRAM recommended)
- 20GB free disk space

## Installation

```bash
# 1. Clone/navigate to the project
cd TXT2SQL

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the package
pip install -e .
```

## Setup API Keys (Optional - for generalist model comparison)

```bash
# Copy the example config
cp config/api_keys.yaml.example config/api_keys.yaml

# Edit config/api_keys.yaml and add your keys:
# - OpenAI API key (for GPT-4)
# - Google API key (for Gemini)
# - Anthropic API key (for Claude)
```

## Download Data

```bash
# Download Spider dataset
python data/scripts/download_spider.py

# Download WikiSQL dataset (optional)
python data/scripts/download_wikisql.py
```

## Train Your First Model

### Option 1: Using Shell Scripts (Recommended)

```bash
# Train Phi-2 with LoRA
bash scripts/train.sh phi-2 lora

# Train Llama-7B with DoRA
bash scripts/train.sh llama-7b dora

# Train Mistral-7B with GRPO
bash scripts/train.sh mistral-7b grpo
```

### Option 2: Using Python Scripts Directly

```bash
# LoRA Training
python experiments/train_lora.py \
    --model microsoft/phi-2 \
    --data_path ./data/spider \
    --output_dir ./results/phi2_lora \
    --epochs 3

# DoRA Training
python experiments/train_dora.py \
    --model meta-llama/Llama-2-7b-hf \
    --output_dir ./results/llama_dora \
    --epochs 3

# GRPO Training
python experiments/train_grpo.py \
    --model mistralai/Mistral-7B-v0.1 \
    --output_dir ./results/mistral_grpo \
    --epochs 5
```

## Evaluate Models

```bash
# Evaluate all models
bash scripts/evaluate.sh 100

# Evaluate only SLMs
bash scripts/evaluate.sh 100 --skip-generalist

# Run comprehensive evaluation
python experiments/evaluate_all.py \
    --data_path ./data/spider \
    --output_dir ./results/evaluation \
    --num_samples 500
```

## Quick Test

Test a trained model on a single query:

```python
from src.models.slm_models import PhiModel
from src.utils.data_loader import SpiderDataset

# Load model
model = PhiModel()
model.load_adapter("./results/phi2_lora")

# Test query
question = "What are all the airlines?"
schema = "CREATE TABLE airlines (airline_id INT, name VARCHAR(100))"
db_id = "flight_db"

sql = model.generate_sql(question, schema, db_id)
print(f"Generated SQL: {sql}")
```

## View Results

```bash
# Open Jupyter notebooks for visualization
jupyter notebook notebooks/

# Check results directory
ls -la results/evaluation/
cat results/evaluation/comparison_table.txt
```

## Common Issues

### CUDA Out of Memory
```bash
# Use smaller batch size
python experiments/train_lora.py --batch_size 2

# Or use gradient accumulation
python experiments/train_lora.py --batch_size 1 --gradient_accumulation_steps 8
```

### Model Download Issues
```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/large/disk

# Or download models manually
huggingface-cli download microsoft/phi-2
```

### API Rate Limits
```bash
# Reduce sample size for generalist models
python experiments/evaluate_all.py --num_samples 50 --generalist_samples 20
```

## Next Steps

1. **Explore Data**: Open `notebooks/01_data_exploration.ipynb`
2. **Analyze Results**: Open `notebooks/02_results_analysis.ipynb`
3. **Customize Training**: Edit `config/training_config.yaml`
4. **Try Different Prompts**: Modify `src/utils/prompt_engineering.py`

## Performance Expectations

| Model | Strategy | Exact Match | Execution Acc | Training Time |
|-------|----------|-------------|---------------|---------------|
| Phi-2 | LoRA | ~60% | ~72% | 2-3 hours |
| Llama-7B | DoRA | ~65% | ~75% | 3-4 hours |
| Mistral-7B | GRPO | ~68% | ~78% | 5-6 hours |
| GPT-4 | - | ~75% | ~85% | N/A |

*Times are approximate on a single A100 GPU*

## Resources

- 📚 [Full Documentation](README.md)
- 🔧 [Configuration Guide](config/README.md)
- 💬 [Report Issues](https://github.com/SaniyaGapchup/txt2sql/issues)
- 📊 [Spider Leaderboard](https://yale-lily.github.io/spider)

## Tips for Best Results

1. **Start Small**: Train on 1000 samples first to verify everything works
2. **Monitor Training**: Use Weights & Biases or TensorBoard
3. **Tune Hyperparameters**: Adjust learning rate and batch size for your GPU
4. **Use Few-Shot Prompts**: Better results with 2-3 examples
5. **Validate SQL**: Always check generated SQL on actual databases

Happy training! 🚀
