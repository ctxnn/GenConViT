# GenConViT: Command Reference Guide

This document provides a comprehensive guide to all commands available for running and evaluating the GenConViT project.

## Table of Contents

1. [Basic Commands](#basic-commands)
2. [Architecture Testing](#architecture-testing)
3. [Performance Comparison](#performance-comparison)
4. [Visualization Tools](#visualization-tools)
5. [Advanced Configuration](#advanced-configuration)

## Basic Commands

### Running Original Prediction

To run a prediction using the original GenConViT architecture:

```bash
python prediction.py --p sample_prediction_data
```

### Running Prediction with Enhanced Architecture (V2)

```bash
python prediction_v2.py --p sample_prediction_data --arch v2
```

### Specifying Number of Frames to Process

```bash
python prediction_v2.py --p sample_prediction_data --f 15
```

### Running on Specific Dataset Types

```bash
# For DFDC dataset
python prediction_v2.py --p DFDC/dfdc_test --d dfdc

# For FaceForensics dataset
python prediction_v2.py --p FaceForensics/data --d faceforensics

# For DeepfakeTIMIT dataset
python prediction_v2.py --p DeepfakeTIMIT --d timit

# For Celeb-DF dataset
python prediction_v2.py --p Celeb-DF --d celeb
```

### Using Half Precision (FP16)

```bash
python prediction_v2.py --p sample_prediction_data --fp16 true
```

## Architecture Testing

### Testing Original Architecture

```bash
python prediction_v2.py --p sample_prediction_data --arch-type original
```

### Testing V2 Architecture (Modified Activations)

```bash
python prediction_v2.py --p sample_prediction_data --arch-type v2
```

### Selecting Network Type

```bash
# For ED network only
python prediction_v2.py --p sample_prediction_data --n ed

# For VAE network only
python prediction_v2.py --p sample_prediction_data --n vae

# For combined networks (default)
python prediction_v2.py --p sample_prediction_data --n genconvit
```

### Specifying Model Size

```bash
# For tiny model
python prediction_v2.py --p sample_prediction_data --s tiny

# For large model
python prediction_v2.py --p sample_prediction_data --s large
```

## Performance Comparison

### Running Complete Benchmark

The benchmark script runs all architecture variants and generates comparison plots:

```bash
./run_benchmark.sh --path sample_prediction_data --frames 15
```

### Customizing Benchmark Parameters

```bash
# Specify custom dataset path
./run_benchmark.sh --path custom_dataset_path

# Specify number of frames to process
./run_benchmark.sh --frames 10
```

### Running Activation Function Comparison

```bash
./compare_activations.sh
```

## Visualization Tools

### Generating Comparison Plots

```bash
python plot_comparison.py --result-dir result --output-dir plots
```

### Filtering Results by Pattern

```bash
# Only compare results from a specific dataset
python plot_comparison.py --pattern "prediction_dfdc_*.json"

# Only compare results from a specific architecture
python plot_comparison.py --pattern "prediction_*_v2_*.json"
```

### Specifying Custom Output Directory

```bash
python plot_comparison.py --output-dir custom_plots_dir
```

## Advanced Configuration

### Specifying Custom Weights

```bash
# Specify ED weights
python prediction_v2.py --p sample_prediction_data --e custom_ed_weights

# Specify VAE weights
python prediction_v2.py --p sample_prediction_data --v custom_vae_weights

# Specify both
python prediction_v2.py --p sample_prediction_data --e custom_ed_weights --v custom_vae_weights
```

### Running Only ED or VAE Component

```bash
# Run only ED component
python prediction_v2.py --p sample_prediction_data --e genconvit_ed_inference

# Run only VAE component
python prediction_v2.py --p sample_prediction_data --v genconvit_vae_inference
```

### Installing Plotting Dependencies

```bash
pip install -r plot_requirements.txt
```

## Command Arguments Reference

### Prediction Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--p` | Video or image path | Required |
| `--f` | Number of frames to process | 15 |
| `--d` | Dataset type (dfdc, faceforensics, timit, celeb) | "other" |
| `--s` | Model size (tiny, large) | - |
| `--e` | Weight file for ED | genconvit_ed_inference |
| `--v` | Weight file for VAE | genconvit_vae_inference |
| `--fp16` | Half precision support | false |
| `--arch` | Architecture type (original, v2) | original |
| `--no-attention` | Disable attention mechanism | false |
| `--no-residual` | Disable residual connections | false |

### Plot Comparison Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--result-dir` | Directory containing result JSON files | result |
| `--output-dir` | Directory to save plots | plots |
| `--pattern` | Pattern to match result files | prediction_*.json |

## Example Workflows

### Complete Testing Workflow

```bash
# Run benchmark on sample data
./run_benchmark.sh --path sample_prediction_data

# Check the plots
open plots/metrics_comparison_*.png

# View the summary report
cat plots/summary_report_*.txt
```

### Custom Testing Workflow

```bash
# Run original architecture
python prediction_v2.py --p sample_prediction_data --arch original

# Run enhanced architecture with specific configuration
python prediction_v2.py --p sample_prediction_data --arch v2 --no-attention

# Generate comparison plots
python plot_comparison.py
```

### Training and Testing Workflow

The project also supports training, though this guide focuses on prediction and evaluation. For training, refer to the original GenConViT documentation.
