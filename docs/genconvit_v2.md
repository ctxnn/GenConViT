# GenConViT V2: Modified Architecture

This document explains the modified version of GenConViT (V2) and how to use it with the existing codebase.

## Architecture Modifications

GenConViT V2 modifies the original GenConViT architecture with the following changes:

1. **Activation Function Changes**: The primary modification in V2 is the replacement of activation functions:
   - **SwiGLU Activation**: Replaces GELU with SwiGLU (Swish-Gated Linear Unit), which can improve gradient flow and model performance.
   - **LeakyReLU Activation**: Replaces standard ReLU with LeakyReLU to prevent dying ReLU problem and potentially improve training dynamics.

2. **Simplified Structure**: 
   - The V2 architecture maintains the same overall structure as the original GenConViT
   - No attention mechanisms or residual connections are used
   - The only differences are the activation functions

3. **Improved Device Handling**:
   - Better management of device placement (CPU/GPU) to prevent runtime errors
   - Ensures tensors are consistently on the same device during inference

These modifications aim to explore the impact of different activation functions on the model's ability to detect deepfakes while maintaining the original model's structure.

## Usage

### Running the Modified Model

To use the GenConViT V2 architecture, you can use the `prediction_v2.py` script which supports both the original and modified architectures:

```bash
python prediction_v2.py --p sample_prediction_data --arch-type v2
```

This will run the modified model with default settings.

### Command Line Arguments

The `prediction_v2.py` script supports the following additional arguments:

- `--arch-type [original|v2]`: Select the architecture type (default: original)
- `--use-attention`: Flag for attention mechanism (ignored in current V2 implementation)
- `--use-residual`: Flag for residual connections (ignored in current V2 implementation)

### Using the Comparison Script

For easier testing and comparison between architectures, you can use the `run_benchmark.sh` script:

```bash
# Run with modified architecture (V2)
./run_benchmark.sh --v2 --path sample_prediction_data

# Run with original architecture
./run_benchmark.sh --original --path sample_prediction_data
```

## Implementation Details

The modified architecture is implemented in `model/genconvit_v2.py` and extends the functionality of the original GenConViT model:

1. The `GenConViTV2` class inherits from `nn.Module` and implements the modified architecture
2. It supports the same modes as the original model (`ed`, `vae`, `genconvit`) 
3. Key modifications include:
   - Custom `SwiGLU` activation function (Swish-Gated Linear Unit)
   - `LeakyReLU` activations to prevent dying ReLU problem
4. The `prediction_v2.py` script provides a unified interface for both architectures

## Benchmarking and Evaluation

When benchmarking this model against the original, you should focus on the impact of activation functions on:

1. Classification performance (accuracy, precision, recall, F1 score)
2. Inference speed
3. Model convergence rate (if retraining)

Use the benchmark scripts (`run_benchmark.sh`) and visualization tools (`plot_comparison.py`) to compare results across different architectures and datasets.

## Switching Between Architectures

If performance does not improve with the new architecture, you can easily switch back to the original one:

1. Using `prediction_v2.py` with `--arch-type original`
2. Using the original `prediction.py` script

## Experimental Results

To compare the results between the original and modified architectures:

1. Run both models on the same dataset
2. Check the results in the `result` directory (files will be named with the architecture type in the filename)
3. Compare accuracy, F1 scores, and other metrics using the `plot_comparison.py` tool

## Next Steps

1. Evaluate performance across different types of deepfakes
2. Consider additional architectural improvements based on experimental results
3. Experiment with other activation functions

## Next Steps

1. Train the enhanced model on the same datasets as the original
2. Evaluate performance across different types of deepfakes
3. Consider additional architectural improvements based on experimental results
