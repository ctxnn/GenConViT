#!/bin/bash

# Script to compare GenConViT architectures with different activation functions

echo "==================================================="
echo "Running comparison between GenConViT architectures"
echo "==================================================="

# Sample data path
SAMPLE_PATH="sample_prediction_data"

# Run the original architecture first
echo -e "\n\n[1/3] Running original GenConViT architecture..."
python prediction_v2.py --p $SAMPLE_PATH --arch original

# Run the new architecture with SwiGLU and LeakyReLU
echo -e "\n\n[2/3] Running GenConViT V2 with SwiGLU and LeakyReLU..."
python prediction_v2.py --p $SAMPLE_PATH --arch v2

# Run the new architecture without attention to isolate effect of activations
echo -e "\n\n[3/3] Running GenConViT V2 with SwiGLU and LeakyReLU (no attention)..."
python prediction_v2.py --p $SAMPLE_PATH --arch v2 --no-attention

echo -e "\n\nComparison complete! Check the results in the result directory."
echo "Files are named with format: prediction_<dataset>_<net>_<arch>_<timestamp>.json"
