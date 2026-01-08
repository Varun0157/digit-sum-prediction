#!/bin/bash

# Script to evaluate all trained models in checkpoints directory

CHECKPOINTS_DIR="checkpoints"
RESULTS_DIR="results"

echo "=========================================="
echo "Evaluating all models in ${CHECKPOINTS_DIR}"
echo "=========================================="
echo ""

# Find all checkpoint files that match the naming pattern
for checkpoint in ${CHECKPOINTS_DIR}/SimpleCNN_k*_*.pth; do
    # Extract filename without path and extension
    filename=$(basename "${checkpoint}" .pth)

    # Skip sanity checkpoints (they're trained on train set)
    if [[ $filename == *"_sanity"* ]]; then
        echo "⏭️  Skipping sanity checkpoint: ${filename}"
        continue
    fi

    # Parse the filename: SimpleCNN_k{kernel}__{pool}_{weight}
    # Example: SimpleCNN_k5_max_balanced
    if [[ $filename =~ SimpleCNN_k([0-9]+)_(max|avg)_(balanced|unweighted) ]]; then
        kernel_size="${BASH_REMATCH[1]}"
        pool_type="${BASH_REMATCH[2]}"
        weight_mode="${BASH_REMATCH[3]}"

        # Construct the eval command
        cmd="uv run -m src.baseline --mode eval --kernel ${kernel_size} --pool ${pool_type}"

        # Add --balance flag if balanced
        if [[ $weight_mode == "balanced" ]]; then
            cmd="${cmd} --balance"
        fi

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📊 Evaluating: ${filename}"
        echo "   kernel=${kernel_size}, pool=${pool_type}, weighted=${weight_mode}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        # Run evaluation
        eval $cmd

        echo ""
        echo "✅ Completed: ${filename}"
        echo ""
    else
        echo "⚠️  Skipping unrecognized format: ${filename}"
    fi
done

echo "=========================================="
echo "✅ All evaluations complete!"
echo "Results saved to: ${RESULTS_DIR}/"
echo "=========================================="
