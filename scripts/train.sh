#!/bin/bash

# Training script for Text-to-SQL models

echo "========================================"
echo "Text-to-SQL Model Training"
echo "========================================"

# Set default values
MODEL=${1:-"phi-2"}
STRATEGY=${2:-"lora"}
EPOCHS=${3:-3}
BATCH_SIZE=${4:-4}

# Map model names to HuggingFace IDs
case $MODEL in
    "phi-2")
        MODEL_ID="microsoft/phi-2"
        ;;
    "llama-7b")
        MODEL_ID="meta-llama/Llama-2-7b-hf"
        ;;
    "mistral-7b")
        MODEL_ID="mistralai/Mistral-7B-v0.1"
        ;;
    *)
        echo "Unknown model: $MODEL"
        echo "Available models: phi-2, llama-7b, mistral-7b"
        exit 1
        ;;
esac

echo "Model: $MODEL ($MODEL_ID)"
echo "Strategy: $STRATEGY"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo ""

# Train based on strategy
case $STRATEGY in
    "lora")
        echo "Training with LoRA..."
        python experiments/train_lora.py \
            --model $MODEL_ID \
            --output_dir ./results/lora/$MODEL \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE
        ;;
    "dora")
        echo "Training with DoRA..."
        python experiments/train_dora.py \
            --model $MODEL_ID \
            --output_dir ./results/dora/$MODEL \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE
        ;;
    "grpo")
        echo "Training with GRPO..."
        python experiments/train_grpo.py \
            --model $MODEL_ID \
            --output_dir ./results/grpo/$MODEL \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE
        ;;
    *)
        echo "Unknown strategy: $STRATEGY"
        echo "Available strategies: lora, dora, grpo"
        exit 1
        ;;
esac

echo ""
echo "✅ Training completed!"
