#!/bin/bash

# Project Setup Verification Script
# This script checks if all components are properly installed and configured

echo "=============================================="
echo "Text-to-SQL Project Setup Verification"
echo "=============================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track issues
ISSUES=0

# Function to check command
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 is installed"
        return 0
    else
        echo -e "${RED}✗${NC} $1 is NOT installed"
        ISSUES=$((ISSUES + 1))
        return 1
    fi
}

# Function to check Python package
check_python_package() {
    if python -c "import $1" &> /dev/null; then
        echo -e "${GREEN}✓${NC} Python package '$1' is installed"
        return 0
    else
        echo -e "${RED}✗${NC} Python package '$1' is NOT installed"
        ISSUES=$((ISSUES + 1))
        return 1
    fi
}

# Function to check file exists
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} File exists: $1"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} File missing: $1"
        return 1
    fi
}

# Function to check directory exists
check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} Directory exists: $1"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} Directory missing: $1"
        return 1
    fi
}

echo "1. Checking System Requirements..."
echo "-----------------------------------"
check_command python
check_command pip
check_command git

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "   Python version: $PYTHON_VERSION"

# Check CUDA availability
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo -e "${GREEN}✓${NC} CUDA is available"
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    echo "   GPU count: $GPU_COUNT"
else
    echo -e "${YELLOW}⚠${NC} CUDA is NOT available (CPU-only mode)"
fi

echo ""
echo "2. Checking Python Dependencies..."
echo "-----------------------------------"
check_python_package torch
check_python_package transformers
check_python_package peft
check_python_package datasets
check_python_package pandas
check_python_package numpy

echo ""
echo "3. Checking Project Structure..."
echo "-----------------------------------"
check_dir "src"
check_dir "src/models"
check_dir "src/utils"
check_dir "src/evaluation"
check_dir "src/training"
check_dir "experiments"
check_dir "config"
check_dir "data"
check_dir "scripts"
check_dir "notebooks"

echo ""
echo "4. Checking Configuration Files..."
echo "-----------------------------------"
check_file "config/model_config.yaml"
check_file "config/training_config.yaml"
check_file "config/api_keys.yaml.example"

if [ -f "config/api_keys.yaml" ]; then
    echo -e "${GREEN}✓${NC} API keys configured"
else
    echo -e "${YELLOW}⚠${NC} API keys not configured (optional for generalist models)"
fi

echo ""
echo "5. Checking Core Source Files..."
echo "-----------------------------------"
check_file "src/models/base_model.py"
check_file "src/models/slm_models.py"
check_file "src/models/generalist_models.py"
check_file "src/utils/data_loader.py"
check_file "src/evaluation/metrics.py"
check_file "src/training/lora_trainer.py"
check_file "src/training/dora_trainer.py"
check_file "src/training/grpo_trainer.py"

echo ""
echo "6. Checking Experiment Scripts..."
echo "-----------------------------------"
check_file "experiments/train_lora.py"
check_file "experiments/train_dora.py"
check_file "experiments/train_grpo.py"
check_file "experiments/evaluate_all.py"

echo ""
echo "7. Checking Shell Scripts..."
echo "-----------------------------------"
check_file "scripts/train.sh"
check_file "scripts/evaluate.sh"

# Make scripts executable
if [ -f "scripts/train.sh" ]; then
    chmod +x scripts/train.sh
fi
if [ -f "scripts/evaluate.sh" ]; then
    chmod +x scripts/evaluate.sh
fi

echo ""
echo "8. Checking Data Directory..."
echo "-----------------------------------"
check_dir "data/scripts"
check_file "data/scripts/download_spider.py"
check_file "data/scripts/download_wikisql.py"

if [ -d "data/spider" ]; then
    echo -e "${GREEN}✓${NC} Spider dataset downloaded"
else
    echo -e "${YELLOW}⚠${NC} Spider dataset not downloaded yet"
    echo "   Run: python data/scripts/download_spider.py"
fi

echo ""
echo "9. Checking Documentation..."
echo "-----------------------------------"
check_file "README.md"
check_file "QUICKSTART.md"
check_file "LICENSE"
check_file "requirements.txt"

echo ""
echo "=============================================="
echo "Verification Complete!"
echo "=============================================="

if [ $ISSUES -eq 0 ]; then
    echo -e "${GREEN}✓ All critical components are properly set up!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Download data: python data/scripts/download_spider.py"
    echo "2. Train a model: bash scripts/train.sh phi-2 lora"
    echo "3. Evaluate: bash scripts/evaluate.sh 100"
    echo ""
    echo "For more details, see QUICKSTART.md"
else
    echo -e "${RED}✗ Found $ISSUES issue(s) that need attention${NC}"
    echo ""
    echo "Please install missing dependencies:"
    echo "  pip install -r requirements.txt"
fi

echo ""
