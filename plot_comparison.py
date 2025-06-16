#!/usr/bin/env python3
import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_result_files(result_dir="result", pattern="prediction_*.json"):
    """
    Load all result JSON files matching pattern from the specified directory
    
    Returns:
        pandas.DataFrame: DataFrame containing metrics from all result files
    """
    result_files = glob.glob(os.path.join(result_dir, pattern))
    results = []
    
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Skip files without metrics
            if 'metrics' not in data:
                print(f"Skipping {file_path} - no metrics found")
                continue
                
            # Extract filename components
            filename = os.path.basename(file_path)
            
            # Create a result entry
            result_entry = {
                'file': filename,
            }
            
            # Add metrics
            if 'metrics' in data:
                for key, value in data['metrics'].items():
                    result_entry[key] = value
            
            # Add metadata if available
            if 'metadata' in data:
                for key, value in data['metadata'].items():
                    if value is not None:  # Skip None values
                        result_entry[key] = value
            
            results.append(result_entry)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    if not results:
        print("No valid result files found")
        return None
        
    return pd.DataFrame(results)

def plot_metrics_comparison(df, output_dir="plots"):
    """
    Generate plots comparing metrics across different runs
    
    Args:
        df: DataFrame containing metrics from all runs
        output_dir: Directory to save the generated plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Accuracy, Precision, Recall, F1-score comparison
    if all(metric in df.columns for metric in ['accuracy', 'precision', 'recall', 'f1_score']):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for plotting
        plot_data = df.melt(
            id_vars=['architecture', 'network'] if all(col in df.columns for col in ['architecture', 'network']) else None,
            value_vars=['accuracy', 'precision', 'recall', 'f1_score'],
            var_name='Metric', value_name='Value'
        )
        
        # Create the grouped bar chart
        sns.barplot(
            data=plot_data, 
            x='architecture' if 'architecture' in plot_data.columns else plot_data.index, 
            y='Value',
            hue='Metric'
        )
        
        plt.title('Performance Metrics Comparison')
        plt.ylabel('Score')
        plt.xlabel('Model Architecture')
        plt.ylim(0, 1.05)  # Metrics are between 0 and 1
        plt.legend(title='Metric')
        plt.tight_layout()
        
        # Save the figure
        fig.savefig(os.path.join(output_dir, f'metrics_comparison_{timestamp}.png'), dpi=300)
        plt.close(fig)
    
    # 2. Confusion matrix-like visualization for each architecture
    if all(metric in df.columns for metric in ['true_fake', 'true_real', 'predicted_fake', 'predicted_real']):
        for idx, row in df.iterrows():
            # Create confusion matrix
            conf_matrix = np.array([
                [row['true_real'] - (row['total_samples'] - row['predicted_fake']), row['total_samples'] - row['predicted_fake'] - row['true_real']],
                [row['predicted_fake'] - row['true_fake'], row['true_fake']]
            ])
            
            # Normalize by dividing by the sum of each row
            conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                conf_matrix_norm, 
                annot=conf_matrix,  # Show raw counts
                fmt='g',
                cmap='Blues',
                xticklabels=['Predicted Real', 'Predicted Fake'],
                yticklabels=['True Real', 'True Fake']
            )
            
            # Title with model info
            title = f"Confusion Matrix"
            if 'architecture' in row and 'network' in row:
                title += f" - {row['architecture']} {row['network']}"
            plt.title(title)
            
            plt.tight_layout()
            
            # Create filename based on available metadata
            filename_parts = []
            for field in ['architecture', 'network', 'dataset']:
                if field in row:
                    filename_parts.append(str(row[field]))
            
            filename = '_'.join(filename_parts) if filename_parts else f"run_{idx}"
            fig.savefig(os.path.join(output_dir, f'confusion_matrix_{filename}_{timestamp}.png'), dpi=300)
            plt.close(fig)
    
    # 3. If we have multiple architectures, compare their performance
    if 'architecture' in df.columns and len(df['architecture'].unique()) > 1:
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            if metric in df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                sns.barplot(
                    data=df, 
                    x='architecture', 
                    y=metric,
                    hue='network' if 'network' in df.columns else None
                )
                
                plt.title(f'{metric.capitalize()} by Architecture')
                plt.ylabel(metric.capitalize())
                plt.xlabel('Architecture')
                plt.ylim(0, 1.05)
                plt.tight_layout()
                
                fig.savefig(os.path.join(output_dir, f'{metric}_by_architecture_{timestamp}.png'), dpi=300)
                plt.close(fig)
    
    # 4. Export metrics to CSV for further analysis
    csv_path = os.path.join(output_dir, f'metrics_summary_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Metrics summary saved to {csv_path}")
    
    # 5. Generate a summary report
    summary_path = os.path.join(output_dir, f'summary_report_{timestamp}.txt')
    with open(summary_path, 'w') as f:
        f.write("GenConViT Performance Comparison Summary\n")
        f.write("======================================\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Model Performance Metrics:\n")
        f.write("--------------------------\n")
        for idx, row in df.iterrows():
            model_name = []
            if 'architecture' in row:
                model_name.append(row['architecture'])
            if 'network' in row:
                model_name.append(row['network'])
            
            model_str = " ".join(model_name) if model_name else f"Model {idx+1}"
            f.write(f"\n{model_str}:\n")
            
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                if metric in row:
                    f.write(f"  {metric.capitalize()}: {row[metric]:.4f}\n")
            
            f.write(f"  Total samples: {row['total_samples'] if 'total_samples' in row else 'N/A'}\n")
            
            # Add other metadata
            if 'runtime_seconds' in row:
                f.write(f"  Runtime: {row['runtime_seconds']:.2f} seconds\n")
    
    print(f"Summary report saved to {summary_path}")
    return os.path.join(output_dir, f'metrics_comparison_{timestamp}.png')

def main():
    parser = argparse.ArgumentParser(description="Plot GenConViT model performance comparison")
    parser.add_argument("--result-dir", default="result", help="Directory containing result JSON files")
    parser.add_argument("--output-dir", default="plots", help="Directory to save plots")
    parser.add_argument("--pattern", default="prediction_*.json", help="Pattern to match result files")
    
    args = parser.parse_args()
    
    print(f"Loading result files from {args.result_dir} matching pattern '{args.pattern}'...")
    df = load_result_files(args.result_dir, args.pattern)
    
    if df is not None:
        print(f"Found {len(df)} result files with metrics")
        output_file = plot_metrics_comparison(df, args.output_dir)
        print(f"Plots generated in {args.output_dir}")
        print(f"Main comparison plot: {output_file}")
    else:
        print("No metrics data found. Make sure you've run prediction with the updated code.")

if __name__ == "__main__":
    main()
