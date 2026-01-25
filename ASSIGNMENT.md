Problem statement: Given an image (and its label value) predict the sum of the digits in the image. Create a preliminary version to act as a baseline using a "standard" CNN, a final version that is your "own" model to compare to the baseline.

You can train using any model (FCNN, CNN, RCNN, etc.)

Note:

- sample code to load data is available in `src/sample/load_data.py`

## Proposed Final Model Approaches (from CLAUDE)

### Multi-Branch CNN
Architecture with multiple parallel convolutional pathways that process input at different scales or with different receptive fields, then merge features through concatenation or attention mechanisms. Inspired by Inception modules and recent hybrid models combining CNN with attention for multi-digit recognition (see: https://www.researchgate.net/publication/379880308_A_Hybrid_Deep_Learning_Model_for_5-Digit_Handwritten_Recognition). Enables simultaneous capture of fine digit details and broader sequence context.

### Vision Transformer (ViT)
Applies transformer architecture to vision tasks by splitting images into patches, embedding them with positional encoding, and processing through self-attention layers. The self-attention mechanism naturally captures relationships between different spatial regions (digit positions) without hard-coded convolutional inductive biases (see: https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html). Vision transformers have become competitive with CNNs across diverse vision tasks including digit recognition.
