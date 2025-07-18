import sys, os
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from time import perf_counter
import pickle
from model.config import load_config
from model.genconvit_ed import GenConViTED
from dataset.loader import load_data, load_checkpoint
import optparse

# Import the fixed VAE model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from genconvit_vae_fixed import GenConViTVAE

config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pretrained(model, optimizer, pretrained_model_filename):
    assert os.path.isfile(
        pretrained_model_filename
    ), "Saved model file does not exist. Exiting."

    model, optimizer, start_epoch, min_loss = load_checkpoint(
        model, optimizer, filename=pretrained_model_filename
    )
    # now individually transfer the optimizer parts...
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    return model, optimizer, start_epoch, min_loss

def train_vae_epoch(model, device, dataloader, optimizer, epoch, criterion, mse_loss):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        classification_output, x_reconstructed, mu, logvar = model(data)
        
        # Calculate losses
        loss_dict = model.get_loss(data, target, classification_output, x_reconstructed, mu, logvar)
        total_loss = loss_dict['total_loss']
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += total_loss.item()
        _, predicted = torch.max(classification_output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} '
                  f'({100. * batch_idx / len(dataloader):.0f}%)]\t'
                  f'Total Loss: {total_loss.item():.6f}\t'
                  f'Recon Loss: {loss_dict["recon_loss"].item():.6f}\t'
                  f'KL Loss: {loss_dict["kl_loss"].item():.6f}\t'
                  f'Cls Loss: {loss_dict["cls_loss"].item():.6f}')
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    print(f'Train Epoch {epoch}: Average Loss: {epoch_loss:.6f}, Accuracy: {epoch_acc:.2f}%')
    return epoch_loss, epoch_acc

def validate_vae_epoch(model, device, dataloader, epoch, criterion, mse_loss):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            classification_output, x_reconstructed, mu, logvar = model(data)
            
            # Calculate losses
            loss_dict = model.get_loss(data, target, classification_output, x_reconstructed, mu, logvar)
            test_loss += loss_dict['total_loss'].item()
            
            # Statistics
            _, predicted = torch.max(classification_output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    test_loss /= len(dataloader)
    test_acc = 100. * correct / total
    
    print(f'Validation Epoch {epoch}: Average Loss: {test_loss:.6f}, Accuracy: {test_acc:.2f}%')
    return test_loss, test_acc

def train_model(dir_path, mod, num_epochs, pretrained_model_filename, test_model, batch_size):
    print("Loading data...")
    dataloaders, dataset_sizes = load_data(dir_path, batch_size)
    print("Done.")
    print(f"Dataset sizes: {dataset_sizes}")
    
    # Initialize model
    if mod == "ed":
        model = GenConViTED(config)
        print("Using GenConViTED model")
    else:
        model = GenConViTVAE(config)
        print("Using GenConViTVAE model")
    
    # Move model to device
    model.to(device)
    print(f"Model moved to device: {device}")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    
    criterion = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    # Load pretrained if specified
    start_epoch = 0
    if pretrained_model_filename:
        model, optimizer, start_epoch, min_loss = load_pretrained(
            model, optimizer, pretrained_model_filename
        )
        print(f"Loaded pretrained model from epoch {start_epoch}")
    
    # Training loop
    torch.manual_seed(1)
    train_losses, train_accs, valid_losses, valid_accs = [], [], [], []
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {num_epochs} epochs...")
    since = time.time()
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        print(f"\nEpoch {epoch+1}/{start_epoch + num_epochs}")
        print("-" * 40)
        
        # Training phase
        if mod == "vae":
            train_loss, train_acc = train_vae_epoch(
                model, device, dataloaders["train"], optimizer, epoch, criterion, mse_loss
            )
            valid_loss, valid_acc = validate_vae_epoch(
                model, device, dataloaders["validation"], epoch, criterion, mse_loss
            )
        else:
            # For ED model, use original training functions
            from train.train_ed import train, valid
            train_losses, train_accs, train_loss = train(
                model, device, dataloaders["train"], criterion, optimizer, epoch,
                train_losses, train_accs, mse_loss
            )
            valid_losses, valid_accs = valid(
                model, device, dataloaders["validation"], criterion, epoch,
                valid_losses, valid_accs, mse_loss
            )
            train_loss = train_losses[-1] if train_losses else 0
            valid_loss = valid_losses[-1] if valid_losses else 0
            train_acc = train_accs[-1] if train_accs else 0
            valid_acc = valid_accs[-1] if valid_accs else 0
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        
        # Step scheduler
        scheduler.step()
        
        # Save best model
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            print(f"New best validation loss: {best_val_loss:.6f}")
    
    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Save model
    print("\nSaving model...")
    file_path = os.path.join(
        "weight",
        f'genconvit_{mod}_{time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime())}',
    )
    
    # Create weight directory if it doesn't exist
    os.makedirs("weight", exist_ok=True)
    
    # Save training history
    with open(f"{file_path}.pkl", "wb") as f:
        pickle.dump([train_losses, train_accs, valid_losses, valid_accs], f)
    
    # Save model state
    state = {
        "epoch": start_epoch + num_epochs,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "min_loss": best_val_loss,
    }
    
    weight_path = f"{file_path}.pth"
    torch.save(state, weight_path)
    print(f"Model saved to: {weight_path}")
    
    if test_model:
        test(model, dataloaders, dataset_sizes, mod, weight_path)

def test(model, dataloaders, dataset_sizes, mod, weight_path):
    print("\nRunning test...")
    model.eval()
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            if mod == "ed":
                outputs = model(inputs)
            else:
                outputs, _, _, _ = model(inputs)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {correct}/{total} ({accuracy:.2f}%)')

def gen_parser():
    parser = optparse.OptionParser("Train GenConViT model.")
    parser.add_option(
        "-e", "--epoch", type=int, dest="epoch",
        help="Number of epochs used for training the GenConvNextViT model."
    )
    parser.add_option("-v", "--version", dest="version", help="Version 0.1.")
    parser.add_option("-d", "--dir", dest="dir", help="Training data path.")
    parser.add_option(
        "-m", "--model", dest="model",
        help="model ed or model vae, model variant: genconvit (A) ed or genconvit (B) vae."
    )
    parser.add_option(
        "-p", "--pretrained", dest="pretrained",
        help="Saved model file name. If you want to continue from the previous trained model."
    )
    parser.add_option("-t", "--test", dest="test", help="run test on test dataset.")
    parser.add_option("-b", "--batch_size", dest="batch_size", help="batch size.")
    
    (options, _) = parser.parse_args()
    
    dir_path = options.dir
    epoch = options.epoch
    mod = "ed" if options.model == "ed" else "vae"
    test_model = options.test == "y"
    pretrained_model_filename = options.pretrained if options.pretrained else None
    batch_size = int(options.batch_size) if options.batch_size else int(config["batch_size"])
    
    return dir_path, mod, epoch, pretrained_model_filename, test_model, batch_size

def main():
    print("GenConViT Training Script")
    print("=" * 50)
    
    start_time = perf_counter()
    path, mod, epoch, pretrained_model_filename, test_model, batch_size = gen_parser()
    
    print(f"Configuration:")
    print(f"  Data path: {path}")
    print(f"  Model: {mod}")
    print(f"  Epochs: {epoch}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")
    print(f"  Pretrained: {pretrained_model_filename}")
    print(f"  Test after training: {test_model}")
    
    try:
        train_model(path, mod, epoch, pretrained_model_filename, test_model, batch_size)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    end_time = perf_counter()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    return 0

if __name__ == "__main__":
    sys.exit(main())