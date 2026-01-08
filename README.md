# Digit Sum Prediction

A CNN-based model for predicting the sum of digits in MNIST images. This project explores different architectural configurations including kernel sizes, pooling strategies, and class weighting approaches.

## Results

Performance of different model configurations on the validation set:

| Kernel Size | Pooling | Weighting | Test Accuracy | Test MAE | Test Loss |
|-------------|---------|-----------|---------------|----------|-----------|
| 3 | Max | Balanced | 26.28% | 1.27 | 2.40 |
| 3 | Max | Unweighted | 25.42% | 1.33 | 2.09 |
| 3 | Avg | Balanced | 28.53% | 1.21 | 2.06 |
| 3 | Avg | Unweighted | 30.08% | 1.17 | 1.96 |
| 5 | Max | Balanced | 36.25% | 0.94 | 2.01 |
| 5 | Max | Unweighted | 38.32% | 0.87 | 2.10 |
| 5 | Avg | Balanced | 52.07% | 0.61 | 1.54 |
| 5 | Avg | Unweighted | 45.22% | 0.73 | 1.58 |
| 7 | Max | Balanced | 28.07% | 1.25 | 2.17 |
| 7 | Max | Unweighted | 34.52% | 0.98 | 1.81 |
| 7 | Avg | Balanced | 56.15% | 0.56 | 2.89 |
| **7** | **Avg** | **Unweighted** | **59.77%** | **0.49** | **1.84** |

**Best Model:** SimpleCNN with kernel size 7, average pooling, and unweighted loss achieves **59.77% accuracy** with **0.49 MAE**.

## Key Findings from Ablation Studies

## Usage

### Training

Train with default configuration:
```bash
uv run -m src.baseline --mode defaults --balance --pool avg
```

Train with different kernel sizes:
```bash
uv run -m src.baseline --mode kernel --balance --pool avg
```

Sanity check (train and validate on training set):
```bash
uv run -m src.baseline --mode sanity --balance --pool avg
```

### Evaluation

Evaluate a trained model:
```bash
uv run -m src.baseline --mode eval --kernel 7 --pool avg
```

Evaluate all trained models:
```bash
bash eval_all.sh
```
