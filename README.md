# Digit Sum Prediction

**Goal**: predict the sum of handwritten digits.

![Sample images](./static/baseline/sample_images.png)

Some exploratory data analysis can be found [here](./data/analysis/).

## Baseline

### Results

Performance of different model configurations on the validation set:

| Kernel Size | Pooling | Weighting      | Test Accuracy | Test MAE |
| ----------- | ------- | -------------- | ------------- | -------- |
| 3           | Max     | Balanced       | 26.28%        | 1.27     |
| 3           | Max     | Unweighted     | 25.42%        | 1.33     |
| 3           | Avg     | Balanced       | 28.53%        | 1.21     |
| 3           | Avg     | Unweighted     | 30.08%        | 1.17     |
| 5           | Max     | Balanced       | 36.25%        | 0.94     |
| 5           | Max     | Unweighted     | 38.32%        | 0.87     |
| 5           | Avg     | Balanced       | 52.07%        | 0.61     |
| 5           | Avg     | Unweighted     | 45.22%        | 0.73     |
| 7           | Max     | Balanced       | 28.07%        | 1.25     |
| 7           | Max     | Unweighted     | 34.52%        | 0.98     |
| 7           | Avg     | Balanced       | 56.15%        | 0.56     |
| **7**       | **Avg** | **Unweighted** | **59.77%**    | **0.49** |

**Best Model:** SimpleCNN with kernel size 7, average pooling, and unweighted loss achieves **59.77% accuracy** with **0.49 MAE**.

#### Confusion Matrix (Best Model)

![Best Model Confusion Matrix](static/baseline/best_model_confusion_matrix.png)

#### Training Plots

The plots for all training runs are available on request, with the sample for all average pooling runs shown below:
| Train Loss | Validation Loss | Validation Accuracy |
| ----- | ----- | ----- |
| ![Train Loss](./static/baseline/wandb-plots/baseline-train-loss.png) | ![Val Loss](./static/baseline/wandb-plots/baseline-val-loss.png) | ![Val Acc](./static/baseline/wandb-plots/baseline-val-acc.png) |

### Key Findings from Ablation Studies

#### Pooling Type

![Pooling Type Comparison](static/baseline/pooling_comparison.png)

Average pooling consistently outperforms max pooling across metrics. This could be because of the spatial invariance it helps bring about.

#### Kernel Size

![Kernel Size Comparison](static/baseline/kernel_comparison.png)

For average pooling, the performance seems to improve as we scale kernel size. This provides us with scope for further testing as well (perhaps we should try kernel sizes of 9, 11, etc.).

For max pooling, the performance caps at a kernel size of 5 and degrades as we move to 7.

![Per Class Kernel Comparison](./static/baseline/perclass_kernel_comparison.png)
Somehow, for rarer sums, we see that a kernel size of 5 occasionally out-performs a kernel size of 7.

#### Class Weighting

![Class Distribution](static/baseline/class_distribution.png)

On performing some exploratory data analysis, we found that the data largely conforms to a Gaussian Distribution. Thus, we tried weighing the classes in a manner inversely proportional to their frequency (capped between 1 and 5) in order to try and boost performance for rarer classes.

![Class Weighting Comparison](static/baseline/weighting_comparison.png)
Still, the unweighted model seems to perform better.

![Per-Class Weighting Comparison](static/baseline/perclass_weighting_comparison.png)
The "balanced" model still seems to bring about some advantages, though. The rarer classes are better represented (as expected) even though the overall performance degrades. If we choose to optimise for a metric that favours these rarer classes, then this could be a useful approach.

**Final Learnings**:

- we should favour average pooling over max pooling
- A kernel size of 7 seems to bring about the best overall performance, but larger kernel sizes may do even better. Also, a kernel size of 5 seems to do better on rare classes. Thus, a multi-branch CNN should be strongly considered for the final model.

### Usage

```bash
uv sync
```

#### Data Preprocessing

Split raw data into train/val sets with stratification:

```bash
uv run -m src.pre.process --data_dir data --output_dir data/processed --val_rat 0.2 --seed 42
```

Analyze the processed data (generates visualizations and quality reports):

```bash
uv run -m src.pre.analyse --data_dir data/processed --output_dir data/analysis --seed 42
```

#### Training

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

#### Evaluation

Evaluate a trained model:

```bash
uv run -m src.baseline --mode eval --kernel 7 --pool avg
```

Evaluate all trained models:

```bash
bash eval_all.sh
```

NOTE: the checkpoints can be found [here](https://drive.google.com/drive/folders/12NJp2T7JPVG_FaXHWO5_D8R8zth-16D6?usp=sharing)

## Main Model

### Hypothesis 1: The Advantages of Multi-Scale Feature Extraction

Based on the experiences from the Baseline model, which saw better performance for Kernel Size 7, but also better performance of kernel size 5 on rare classes, we decided to build a multi-branch CNN that extracts features at multiple scales to combine the advantages of each kernel size.

However, the results were disappointing, with us barely breaking more than 1% over baseline performance.

> > > some results from this model

### Hypothesis 2: Digit Prediction is Easier than Sum Prediction

Prediction of each digit is fundamentally a much simpler task than prediction of the final sum. But, the labels we've been provided only contain the final sum and not the individual digits.

Thus, we aim to extract digit level labels from our samples.

#### Data Extraction

##### Attempt 1: OCR

Using both `tesseractt` and `easyocr` led to underwhelming results. On extracting digits (with and without colour inversion), we could never break about 50% success, where a successful extraction is one where the sum of predicted digits equals the ground truth sum.

Thus, we had to be more creative.

##### Attempt 2: Pre-trained MNIST + self-labelling

Using the magic of digital image processing, we apply the following to each image among our provided samples:

- contour detection
- within each contour:
  > > > show initial image
  - erode the digit
  - pad with black
    > > > show final image and MNIST counter-part
  - classify\*
- add all of the predictions and compare with the ground truth sum. If the sum is correct, the extraction is assumed to have been successful.

* - we create a simple conv-net and pre-train it on MNIST.

On applying the above pipeline, we were successfully able to extract \_\_\_\_ samples.

At this stage, we build our preliminary model - ResNet feature extraction followed by four classification heads that predict each digit.

We then apply the following pipeline repeatedly to self-label:

- train on available data
- attempt to classify remaining data
  - for samples where predicted sum is correct, add to labelled set

After applying the above 7 times, we only had about 250 remaining unlabelled samples. In order to classify these, we manually labelled most of them. The remaining (12) unlabelled samples were quite hard even for me to make sense of, so I left them unlabelled and part of the test set.

> > > add picture of manual labelling GUI

Now, we have a digit labelled corpus of samples and their corresponding digits with decent confidence.

#### Modelling

Since it is usually adept at feature extraction, we use a ResNet based backbone with some simple dense classification heads.

Immediately, we see a significant boost in performance. With just about a million parameters (half of the best baseline model) we get a test accuracy of about 92.63%.

##### Experiments

###### Deeper and Wider

![Width Comparison](./static/main/width_comparison.png)

###### Initial Kernel Sizes

![Kernel Comparison](./static/main/kernel_comparison.png)

###### Regularising Using Total Sum

![Sum Loss Comparison](./static/main/sumloss_comparison.png)

###### Spatial Attention

###### Augmentation

![Augmentation Comparison](./static/main/augmentation_comparison.png)

###### Summary

| Config | Val Sum Acc | Test Sum Acc | Val Digit Acc | Params |
|--------|-------------|--------------|---------------|--------|
| **aug** | **94.63%** | **93.63%** | **97.25%** | 1.22M |
| k5 | 93.40% | 93.07% | 96.92% | 1.22M |
| w1.50 | 93.33% | 92.93% | 96.93% | 2.96M |
| sum0.5 | 93.33% | 92.43% | 96.93% | 1.22M |
| baseline (k7) | 93.07% | 92.63% | 96.87% | 1.22M |
| sum1.0 | 92.90% | 92.33% | 96.81% | 1.22M |
| w1.25 | 92.13% | 91.03% | 96.63% | 2.06M |
| k3 | 91.67% | 91.97% | 96.50% | 1.22M |

# TODO

- [ ] clean up and un-gpt
- [ ] remove test OCR models from uv
