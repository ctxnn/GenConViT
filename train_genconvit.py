"""
Simple Training Script for GenConViT
-----------------------------------
A simplified training script for GenConViT models that can work independently.

Usage:
    python train_genconvit.py --data_dir /path/to/data --model_type ed --epochs 10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import os
import time
import json
from genconvit_standalone import create_genconvit, load_config


def get_data_transforms():
    """Get data transformations for training and validation"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    return {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }


def load_datasets(data_dir, batch_size=32):
    """Load training and validation datasets"""
    transforms_dict = get_data_transforms()
    
    # Expected structure: data_dir/train/{real,fake} and data_dir/val/{real,fake}
    datasets_dict = {
        'train': datasets.ImageFolder(
            os.path.join(data_dir, 'train'),
            transform=transforms_dict['train']
        ),
        'val': datasets.ImageFolder(
            os.path.join(data_dir, 'val'),
            transform=transforms_dict['val']
        )
    }
    
    dataloaders = {
        'train': DataLoader(
            datasets_dict['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        ),
        'val': DataLoader(
            datasets_dict['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    }
    
    dataset_sizes = {x: len(datasets_dict[x]) for x in ['train', 'val']}
    
    return dataloaders, dataset_sizes


def train_epoch(model, dataloader, criterion, optimizer, device, model_type):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if model_type == 'vae':
            outputs, reconstructed = model.model_vae(inputs)
            # Add reconstruction loss for VAE
            mse_loss = nn.MSELoss()(reconstructed, inputs)
            classification_loss = criterion(outputs, labels)
            loss = classification_loss + 0.1 * mse_loss  # Weight reconstruction loss
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device, model_type):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            if model_type == 'vae':
                outputs, reconstructed = model.model_vae(inputs)
                # Add reconstruction loss for VAE
                mse_loss = nn.MSELoss()(reconstructed, inputs)
                classification_loss = criterion(outputs, labels)
                loss = classification_loss + 0.1 * mse_loss
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc


def train_model(data_dir, model_type='ed', architecture='original', epochs=10, 
                batch_size=32, learning_rate=0.0001, save_dir='./checkpoints'):
    """Main training function"""
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load configuration and create model
    config = load_config()
    
    try:
        model = create_genconvit(model_type, architecture, config, pretrained=True)
        model = model.to(device)
        print(f"Created {architecture} GenConViT model (type: {model_type})")
    except Exception as e:
        print(f"Error creating model: {e}")
        print("Falling back to simple CNN architecture...")
        # Fallback to a simple architecture if timm models fail
        config['model']['latent_dims'] = 768  # Reduced latent dims
        model = create_genconvit(model_type, 'v2', config, pretrained=False)
        model = model.to(device)
    
    # Load datasets
    try:
        dataloaders, dataset_sizes = load_datasets(data_dir, batch_size)
        print(f"Loaded datasets - Train: {dataset_sizes['train']}, Val: {dataset_sizes['val']}")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Please ensure your data directory has the structure: data_dir/train/{real,fake} and data_dir/val/{real,fake}")
        return
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_acc = 0.0
    best_model_path = os.path.join(save_dir, f'best_{model_type}_{architecture}.pth')
    
    print(f"Starting training for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        print('-' * 10)
        
        # Training phase
        train_loss, train_acc = train_epoch(
            model, dataloaders['train'], criterion, optimizer, device, model_type
        )
        
        # Validation phase
        val_loss, val_acc = validate_epoch(
            model, dataloaders['val'], criterion, device, model_type
        )
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.item())
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())
        
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': config,
                'model_type': model_type,
                'architecture': architecture
            }, best_model_path)
            print(f'New best model saved with accuracy: {best_acc:.4f}')
    
    # Save final model and history
    final_model_path = os.path.join(save_dir, f'final_{model_type}_{architecture}.pth')
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_acc': val_acc,
        'config': config,
        'model_type': model_type,
        'architecture': architecture
    }, final_model_path)
    
    # Save training history
    history_path = os.path.join(save_dir, f'history_{model_type}_{architecture}.json')
    with open(history_path, 'w') as f:
        # Convert tensors to floats for JSON serialization
        history_serializable = {
            key: [float(val) if torch.is_tensor(val) else val for val in values]
            for key, values in history.items()
        }
        json.dump(history_serializable, f, indent=2)
    
    training_time = time.time() - start_time
    print(f'\nTraining completed in {training_time//60:.0f}m {training_time%60:.0f}s')
    print(f'Best validation accuracy: {best_acc:.4f}')
    print(f'Models saved to: {save_dir}')


def main():
    parser = argparse.ArgumentParser(description='Train GenConViT for deepfake detection')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory (should contain train/ and val/ subdirs)')
    parser.add_argument('--model_type', choices=['ed', 'vae', 'both'], default='ed',
                        help='Model type to train')
    parser.add_argument('--architecture', choices=['original', 'v2'], default='original',
                        help='Architecture variant')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    
    # Validate data directory
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist")
        return
    
    train_model(
        data_dir=args.data_dir,
        model_type=args.model_type,
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
