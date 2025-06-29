"""
Simple Evaluation Script for GenConViT
-------------------------------------
A simplified evaluation script for GenConViT models.

Usage:
    python evaluate_genconvit.py --data_dir /path/to/test_data --model_path /path/to/model.pth
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import os
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from genconvit_standalone import create_genconvit, load_config


def get_test_transforms():
    """Get data transformations for testing"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def load_test_dataset(data_dir, batch_size=32):
    """Load test dataset"""
    transform = get_test_transforms()
    
    # Expected structure: data_dir/test/{real,fake} or data_dir/{real,fake}
    test_dir = os.path.join(data_dir, 'test') if os.path.exists(os.path.join(data_dir, 'test')) else data_dir
    
    dataset = datasets.ImageFolder(test_dir, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader, len(dataset), dataset.classes


def load_model(model_path, device):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model configuration
    config = checkpoint.get('config', load_config())
    model_type = checkpoint.get('model_type', 'ed')
    architecture = checkpoint.get('architecture', 'original')
    
    print(f"Loading {architecture} {model_type} model...")
    
    # Create model
    try:
        model = create_genconvit(model_type, architecture, config, pretrained=False)
    except Exception as e:
        print(f"Error creating model with saved config: {e}")
        print("Trying with default config...")
        config = load_config()
        config['model']['latent_dims'] = 768
        model = create_genconvit(model_type, 'v2', config, pretrained=False)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, model_type, architecture


def evaluate_model(model, dataloader, device, model_type):
    """Evaluate model on test dataset"""
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    total_correct = 0
    total_samples = 0
    
    print("Evaluating model...")
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            if model_type == 'vae':
                outputs, _ = model.model_vae(inputs)
            else:
                outputs = model(inputs)
            
            # Get predictions
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            # Collect results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Calculate accuracy
            total_correct += torch.sum(predictions == labels).item()
            total_samples += labels.size(0)
            
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx * len(inputs)}/{len(dataloader.dataset)} samples")
    
    accuracy = total_correct / total_samples
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities), accuracy


def calculate_metrics(predictions, labels, probabilities, class_names):
    """Calculate various evaluation metrics"""
    
    # Basic metrics
    accuracy = np.mean(predictions == labels)
    
    # Classification report
    report = classification_report(labels, predictions, target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # ROC AUC (for binary classification)
    if len(class_names) == 2:
        try:
            auc = roc_auc_score(labels, probabilities[:, 1])
        except:
            auc = None
    else:
        auc = None
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'auc': auc
    }


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_classification_metrics(report, save_path=None):
    """Plot classification metrics"""
    # Extract metrics for each class
    classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    
    metrics = ['precision', 'recall', 'f1-score']
    values = {metric: [report[cls][metric] for cls in classes] for metric in metrics}
    
    # Create subplot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(classes))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, values[metric], width, label=metric.title())
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Score')
    ax.set_title('Classification Metrics by Class')
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, metric in enumerate(metrics):
        for j, v in enumerate(values[metric]):
            ax.text(j + i * width, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Classification metrics plot saved to {save_path}")
    
    plt.show()


def save_results(metrics, predictions, labels, probabilities, save_dir, model_info):
    """Save evaluation results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(save_dir, 'evaluation_metrics.json')
    
    # Make metrics JSON serializable
    serializable_metrics = {
        'accuracy': float(metrics['accuracy']),
        'auc': float(metrics['auc']) if metrics['auc'] is not None else None,
        'classification_report': metrics['classification_report'],
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
        'model_info': model_info
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    # Save detailed results
    results_path = os.path.join(save_dir, 'detailed_results.npz')
    np.savez(results_path, 
             predictions=predictions, 
             labels=labels, 
             probabilities=probabilities)
    
    print(f"Results saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate GenConViT for deepfake detection')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to test dataset directory')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--save_dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--plot', action='store_true',
                        help='Generate and save plots')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Validate inputs
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist")
        return
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint {args.model_path} does not exist")
        return
    
    try:
        # Load test dataset
        dataloader, dataset_size, class_names = load_test_dataset(args.data_dir, args.batch_size)
        print(f"Loaded test dataset with {dataset_size} samples")
        print(f"Classes: {class_names}")
        
        # Load model
        model, model_type, architecture = load_model(args.model_path, device)
        
        # Evaluate model
        predictions, labels, probabilities, accuracy = evaluate_model(model, dataloader, device, model_type)
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, labels, probabilities, class_names)
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        if metrics['auc'] is not None:
            print(f"AUC Score: {metrics['auc']:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(labels, predictions, target_names=class_names))
        
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        
        # Model info
        model_info = {
            'model_type': model_type,
            'architecture': architecture,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        print(f"\nModel Info:")
        print(f"Type: {model_info['model_type']}")
        print(f"Architecture: {model_info['architecture']}")
        print(f"Total Parameters: {model_info['total_parameters']:,}")
        print(f"Trainable Parameters: {model_info['trainable_parameters']:,}")
        
        # Save results
        save_results(metrics, predictions, labels, probabilities, args.save_dir, model_info)
        
        # Generate plots if requested
        if args.plot:
            try:
                cm_path = os.path.join(args.save_dir, 'confusion_matrix.png')
                plot_confusion_matrix(metrics['confusion_matrix'], class_names, cm_path)
                
                metrics_path = os.path.join(args.save_dir, 'classification_metrics.png')
                plot_classification_metrics(metrics['classification_report'], metrics_path)
            except Exception as e:
                print(f"Error generating plots: {e}")
                print("Plots require matplotlib and seaborn")
        
        print(f"\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
