# GenConViT Performance Comparison Tools

This document explains how to use the performance comparison tools to evaluate different GenConViT architectures.

## Overview

The enhanced prediction system and visualization tools allow you to:

1. Compare performance metrics (accuracy, precision, recall, F1-score) across different architecture variants
2. Generate visual comparisons through plots and charts
3. Export detailed metrics to CSV for further analysis
4. Create a summary report of all runs

## Running the Benchmark

The simplest way to compare architectures is to use the provided benchmark script:

```bash
./run_benchmark.sh --path sample_prediction_data --frames 15
```

This will:
1. Run the original GenConViT architecture
2. Run the GenConViT V2 with all enhancements
3. Run the GenConViT V2 with no attention mechanism
4. Run the GenConViT V2 with no residual connections
5. Generate comparison plots and reports

## Manual Comparison

If you want more control over the comparison process, you can run each test manually:

```bash
# Run original architecture
python prediction_v2.py --p sample_prediction_data --arch original

# Run v2 architecture
python prediction_v2.py --p sample_prediction_data --arch v2

# Then generate plots
python plot_comparison.py
```

## Visualization Options

The `plot_comparison.py` script provides several visualization options:

```bash
python plot_comparison.py --result-dir result --output-dir plots --pattern "prediction_*.json"
```

Arguments:
- `--result-dir`: Directory containing result JSON files (default: "result")
- `--output-dir`: Directory to save plots (default: "plots")
- `--pattern`: Pattern to match result files (default: "prediction_*.json")

## Output Files

The visualization tool generates the following outputs:

1. **Metrics Comparison Plot**: Bar chart comparing accuracy, precision, recall, and F1-score
2. **Confusion Matrix Plots**: Heatmap visualizations of confusion matrices for each run
3. **Individual Metric Plots**: Bar charts for each metric across architectures
4. **CSV Summary**: Detailed metrics in CSV format for further analysis
5. **Text Summary Report**: Human-readable summary of all comparison results

## Dependencies

To use the plotting functionality, install the required dependencies:

```bash
pip install -r plot_requirements.txt
```

## JSON Format

The enhanced prediction system outputs JSON files with the following structure:

```json
{
  "metrics": {
    "accuracy": 0.95,
    "precision": 0.92,
    "recall": 0.98,
    "f1_score": 0.95,
    "total_samples": 100,
    "true_fake": 50,
    "true_real": 50,
    "predicted_fake": 52,
    "predicted_real": 48
  },
  "metadata": {
    "architecture": "v2",
    "network": "genconvit",
    "dataset": "sample_data",
    "frames_processed": 15,
    "use_attention": true,
    "use_residual": true,
    "runtime_seconds": 120.5
  },
  "video": {
    // Original video result data
  }
}
```
