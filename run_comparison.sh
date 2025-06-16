#!/bin/bash

# Script to test the new GenConViT V2 architecture on sample data

# Set default parameters
DATASET_PATH="sample_prediction_data"
FRAMES=15
ARCH="v2"  # Use the new architecture
USE_ATTENTION=true
USE_RESIDUAL=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --original)
      ARCH="original"
      shift
      ;;
    --v2)
      ARCH="v2"
      shift
      ;;
    --no-attention)
      NO_ATTENTION="--no-attention"
      shift
      ;;
    --no-residual)
      NO_RESIDUAL="--no-residual"
      shift
      ;;
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

# Print configuration
echo "Running GenConViT prediction with the following settings:"
echo "Architecture: $ARCH"
echo "Dataset path: $DATASET_PATH"
echo "Frames to process: $FRAMES"

# Construct command
CMD="python prediction_v2.py --p $DATASET_PATH --f $FRAMES --arch $ARCH"

# Add optional flags
if [[ -n "$NO_ATTENTION" ]]; then
  CMD="$CMD --no-attention"
  echo "Attention mechanism: Disabled"
else
  echo "Attention mechanism: Enabled"
fi

if [[ -n "$NO_RESIDUAL" ]]; then
  CMD="$CMD --no-residual"
  echo "Residual connections: Disabled"
else
  echo "Residual connections: Enabled"
fi

# Execute command
echo "Executing: $CMD"
echo "-------------------------------------------"
eval $CMD
