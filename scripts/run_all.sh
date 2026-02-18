#!/bin/bash
# Run the full mining pipeline: generate -> evaluate -> analyze

set -e

echo "=========================================="
echo "Mining LLMs for Low-Resource Language Data"
echo "=========================================="

echo ""
echo "Step 1: Generating outputs..."
echo "---"
python scripts/run_generation.py --all

echo ""
echo "Step 2: Evaluating outputs..."
echo "---"
python scripts/run_evaluation.py --all

echo ""
echo "Step 3: Analyzing results..."
echo "---"
python scripts/run_analysis.py

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "Results saved in: results/"
echo "=========================================="
