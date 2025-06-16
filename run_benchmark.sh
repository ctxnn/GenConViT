#!/bin/bash

# Script to benchmark different GenConViT architectures and create comparison plots

echo "================================================================"
echo "Running GenConViT Architecture Benchmark"
echo "================================================================"

# Default parameters
DATASET_PATH="sample_prediction_data"
FRAMES=15

# Make plots directory if it doesn't exist
mkdir -p plots

# Check if the results directory exists
mkdir -p result

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --path)
      DATASET_PATH="$2"
      shift 2
      ;;
    --frames)
      FRAMES="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Print benchmark configuration
echo "Running benchmark with the following settings:"
echo "Dataset path: $DATASET_PATH"
echo "Frames to process: $FRAMES"
echo "================================================================"

# Run tests with different architectures
echo -e "\n[1/2] Testing original GenConViT architecture..."
python prediction_v2.py --p "$DATASET_PATH" --f "$FRAMES" --arch-type original

echo -e "\n[2/2] Testing GenConViT V2 architecture (modified activations)..."
python prediction_v2.py --p "$DATASET_PATH" --f "$FRAMES" --arch-type v2

echo -e "\n\nGenerating comparison plots..."
python plot_comparison.py --result-dir result --output-dir plots

echo -e "\nBenchmark complete! Check the plots directory for visual comparisons."
echo "----------------------------------------------------------------"
