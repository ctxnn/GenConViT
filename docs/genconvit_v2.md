# GenConViT V2: Enhanced Architecture

This document explains the enhanced version of GenConViT (V2) and how to use it with the existing codebase.

## Architecture Enhancements

GenConViT V2 builds upon the original GenConViT architecture with the following enhancements:

1. **Cross-Attention Mechanism**: Adds attention between the Encoder-Decoder (ED) and Variational Autoencoder (VAE) branches, allowing the model to focus on more discriminative features.

2. **Feature Fusion Layer with SwiGLU**: Improved integration of features from both branches with a dedicated fusion module using SwiGLU activation (Swish-Gated Linear Unit), which can improve gradient flow and model performance.

3. **LeakyReLU Activation**: Replaces standard ReLU with LeakyReLU to prevent dying ReLU problem and potentially improve training dynamics.

4. **Optional Residual Connections**: Support for residual connections to help maintain important information throughout the network.

These enhancements aim to improve the model's ability to detect deepfakes by better leveraging the complementary information from both branches of the network.

## Usage

### Running the Enhanced Model

To use the enhanced GenConViT V2 architecture, you can use the `prediction_v2.py` script which supports both the original and enhanced architectures:

```bash
python prediction_v2.py --p sample_prediction_data --arch v2
```

This will run the enhanced model with default settings (attention and residual connections enabled).

### Command Line Arguments

The `prediction_v2.py` script supports the following additional arguments:

- `--arch [original|v2]`: Select the architecture type (default: original)
- `--no-attention`: Disable the attention mechanism for V2 architecture
- `--no-residual`: Disable residual connections for V2 architecture

### Using the Comparison Script

For easier testing and comparison between architectures, you can use the `run_comparison.sh` script:

```bash
# Run with enhanced architecture (V2)
./run_comparison.sh --v2 --path sample_prediction_data

# Run with original architecture
./run_comparison.sh --original --path sample_prediction_data

# Disable specific features of V2
./run_comparison.sh --v2 --no-attention --path sample_prediction_data
./run_comparison.sh --v2 --no-residual --path sample_prediction_data
```

## Implementation Details

The enhanced architecture is implemented in `model/genconvit_v2.py` and extends the functionality of the original GenConViT model:

1. The `GenConViTV2` class inherits from `nn.Module` and implements the enhanced architecture
2. It supports the same modes as the original model (`ed`, `vae`, `genconvit`) plus a new `v2` mode
3. Key improvements include:
   - Custom `SwiGLU` activation function (Swish-Gated Linear Unit) for feature fusion
   - `LeakyReLU` activations to prevent dying ReLU problem
   - Cross-attention mechanism between ED and VAE branches
   - Optional residual connections
4. The `prediction_v2.py` script provides a unified interface for both architectures

## Switching Between Architectures

If performance does not improve with the new architecture, you can easily switch back to the original one:

1. Using `prediction_v2.py` with `--arch original`
2. Using the original `prediction.py` script

## Experimental Results

To compare the results between the original and enhanced architectures:

1. Run both models on the same dataset
2. Check the results in the `result` directory (files will be named with the architecture type in the filename)
3. Compare accuracy, F1 scores, and other metrics

## Next Steps

1. Train the enhanced model on the same datasets as the original
2. Evaluate performance across different types of deepfakes
3. Consider additional architectural improvements based on experimental results
