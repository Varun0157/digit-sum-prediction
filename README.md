# Digit Sum Prediction

A CNN-based model for predicting the sum of digits in MNIST images. This project explores different architectural configurations including kernel sizes, pooling strategies, and class weighting approaches.

## Results

Performance of different model configurations on the validation set:

| Kernel | Test Accuracy (%) |  |  |  | Test MAE |  |  |  |
|--------|----------|----------|----------|----------|------|------|------|------|
|        | **Max-B** | **Max-U** | **Avg-B** | **Avg-U** | **Max-B** | **Max-U** | **Avg-B** | **Avg-U** |
| **3**  | 26.28 | 25.42 | 28.53 | 30.08 | 1.27 | 1.33 | 1.21 | 1.17 |
| **5**  | 36.25 | 38.32 | 52.07 | 45.22 | 0.94 | 0.87 | 0.61 | 0.73 |
| **7**  | 28.07 | 34.52 | 56.15 | **59.77** | 1.25 | 0.98 | 0.56 | **0.49** |

*B = Balanced (class-weighted), U = Unweighted*

**Best Model:** Kernel size 7, average pooling, unweighted loss → **59.77% accuracy, 0.49 MAE**

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
