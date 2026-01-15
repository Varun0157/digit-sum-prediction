#!/bin/bash
# Multi-head ResNet ablation study
# Ablations: kernel_size, sum_loss_weight, width_multiplier
# All runs: 200 epochs, patience 20

set -e  # Exit on error

EPOCHS=200
PATIENCE=20
BATCH_SIZE=128
LR=1e-3

echo "=============================================="
echo "MULTI-HEAD RESNET ABLATION STUDY"
echo "=============================================="

# -----------------------------------------------------------------------------
# TRAINING
# -----------------------------------------------------------------------------

echo ""
echo "=== TRAINING ==="
echo ""

# 1. Baseline (k7, sum0, w1.0)
echo "[1/7] Training baseline (k7, sum0, w1.0)..."
uv run python -m src.train_multihead --epochs $EPOCHS --patience $PATIENCE --batch_size $BATCH_SIZE --lr $LR

# 2. Kernel size = 3
echo "[2/7] Training kernel_size=3..."
uv run python -m src.train_multihead --epochs $EPOCHS --patience $PATIENCE --batch_size $BATCH_SIZE --lr $LR --kernel_size 3

# 3. Kernel size = 5
echo "[3/7] Training kernel_size=5..."
uv run python -m src.train_multihead --epochs $EPOCHS --patience $PATIENCE --batch_size $BATCH_SIZE --lr $LR --kernel_size 5

# 4. Sum loss weight = 0.5
echo "[4/7] Training sum_loss_weight=0.5..."
uv run python -m src.train_multihead --epochs $EPOCHS --patience $PATIENCE --batch_size $BATCH_SIZE --lr $LR --sum_loss_weight 0.5

# 5. Sum loss weight = 1.0
echo "[5/7] Training sum_loss_weight=1.0..."
uv run python -m src.train_multihead --epochs $EPOCHS --patience $PATIENCE --batch_size $BATCH_SIZE --lr $LR --sum_loss_weight 1.0

# 6. Width multiplier = 1.25
echo "[6/7] Training width_multiplier=1.25..."
uv run python -m src.train_multihead --epochs $EPOCHS --patience $PATIENCE --batch_size $BATCH_SIZE --lr $LR --width_multiplier 1.25

# 7. Width multiplier = 1.5
echo "[7/7] Training width_multiplier=1.5..."
uv run python -m src.train_multihead --epochs $EPOCHS --patience $PATIENCE --batch_size $BATCH_SIZE --lr $LR --width_multiplier 1.5

# -----------------------------------------------------------------------------
# EVALUATION (Validation Set - includes digit accuracy)
# -----------------------------------------------------------------------------

echo ""
echo "=== EVALUATION ON VALIDATION SET ==="
echo ""

# 1. Baseline
echo "[1/7] Evaluating baseline on val..."
uv run python -m src.test_multihead --checkpoint checkpoints/multihead_resnet_best.pth --split val

# 2. Kernel size = 3
echo "[2/7] Evaluating k3 on val..."
uv run python -m src.test_multihead --checkpoint checkpoints/multihead_resnet_best_k3.pth --kernel_size 3 --split val

# 3. Kernel size = 5
echo "[3/7] Evaluating k5 on val..."
uv run python -m src.test_multihead --checkpoint checkpoints/multihead_resnet_best_k5.pth --kernel_size 5 --split val

# 4. Sum loss weight = 0.5
echo "[4/7] Evaluating sum0.5 on val..."
uv run python -m src.test_multihead --checkpoint checkpoints/multihead_resnet_best_sum05.pth --split val

# 5. Sum loss weight = 1.0
echo "[5/7] Evaluating sum1.0 on val..."
uv run python -m src.test_multihead --checkpoint checkpoints/multihead_resnet_best_sum10.pth --split val

# 6. Width multiplier = 1.25
echo "[6/7] Evaluating w1.25 on val..."
uv run python -m src.test_multihead --checkpoint checkpoints/multihead_resnet_best_w125.pth --width_multiplier 1.25 --split val

# 7. Width multiplier = 1.5
echo "[7/7] Evaluating w1.5 on val..."
uv run python -m src.test_multihead --checkpoint checkpoints/multihead_resnet_best_w150.pth --width_multiplier 1.5 --split val

# -----------------------------------------------------------------------------
# TESTING (Test Set - sum accuracy only)
# -----------------------------------------------------------------------------

echo ""
echo "=== TESTING ON TEST SET ==="
echo ""

# 1. Baseline
echo "[1/7] Testing baseline..."
uv run python -m src.test_multihead --checkpoint checkpoints/multihead_resnet_best.pth --split test

# 2. Kernel size = 3
echo "[2/7] Testing k3..."
uv run python -m src.test_multihead --checkpoint checkpoints/multihead_resnet_best_k3.pth --kernel_size 3 --split test

# 3. Kernel size = 5
echo "[3/7] Testing k5..."
uv run python -m src.test_multihead --checkpoint checkpoints/multihead_resnet_best_k5.pth --kernel_size 5 --split test

# 4. Sum loss weight = 0.5
echo "[4/7] Testing sum0.5..."
uv run python -m src.test_multihead --checkpoint checkpoints/multihead_resnet_best_sum05.pth --split test

# 5. Sum loss weight = 1.0
echo "[5/7] Testing sum1.0..."
uv run python -m src.test_multihead --checkpoint checkpoints/multihead_resnet_best_sum10.pth --split test

# 6. Width multiplier = 1.25
echo "[6/7] Testing w1.25..."
uv run python -m src.test_multihead --checkpoint checkpoints/multihead_resnet_best_w125.pth --width_multiplier 1.25 --split test

# 7. Width multiplier = 1.5
echo "[7/7] Testing w1.5..."
uv run python -m src.test_multihead --checkpoint checkpoints/multihead_resnet_best_w150.pth --width_multiplier 1.5 --split test

echo ""
echo "=============================================="
echo "ABLATION STUDY COMPLETE"
echo "=============================================="
echo "Results saved in results/ directory"
