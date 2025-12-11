#!/bin/bash

# Evaluation script for Text-to-SQL models

echo "========================================"
echo "Text-to-SQL Model Evaluation"
echo "========================================"

# Parse command line arguments
NUM_SAMPLES=${1:-100}
SKIP_SLM=${2:-false}
SKIP_GENERALIST=${3:-false}

echo "Number of samples: $NUM_SAMPLES"
echo "Skip SLM: $SKIP_SLM"
echo "Skip Generalist: $SKIP_GENERALIST"
echo ""

# Build command
CMD="python experiments/evaluate_all.py --num_samples $NUM_SAMPLES"

if [ "$SKIP_SLM" = "true" ]; then
    CMD="$CMD --skip_slm"
fi

if [ "$SKIP_GENERALIST" = "true" ]; then
    CMD="$CMD --skip_generalist"
fi

# Run evaluation
echo "Running evaluation..."
eval $CMD

echo ""
echo "✅ Evaluation completed!"
echo "Results saved to: ./results/evaluation"
