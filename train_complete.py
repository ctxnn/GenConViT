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
from dataset.loader import load_data, load_checkpoint
import optparse

# Import the fixed models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from genconvit_vae_final import GenConViTVAE
from genconvit_ed_final import GenConViTED

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
    running_recon_loss = 0.0
    running_kl_loss = 0.0
    running_cls_loss = 0.0
    correct = 0
    total = 0
    
    print(f"\nStarting VAE training epoch {epoch}")
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        try:
            # Forward pass
            classification_output, x_reconstructed, mu, logvar = model(data)
            
            # Calculate losses
            loss_dict = model.get_loss(data, target, classification_output, x_reconstructed, mu, logvar)
            total_loss = loss_dict['total_loss']
            
            # Check for NaN losses
            if torch.isnan(total_loss):
                print(f"NaN loss detected in batch {batch_idx}, skipping...")
                continue
                
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            running_loss += total_loss.item()
            running_recon_loss += loss_dict['recon_loss'].item()
            running_kl_loss += loss_dict['kl_loss'].item()
            running_cls_loss += loss_dict['cls_loss'].item()
            
            _, predicted = torch.max(classification_output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 50 == 0:
                current_acc = 100. * correct / total if total > 0 else 0
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} '
                      f'({100. * batch_idx / len(dataloader):.0f}%)]\t'
                      f'Loss: {total_loss.item():.6f}\t'
                      f'Recon: {loss_dict["recon_loss"].item():.4f}\t'
                      f'KL: {loss_dict["kl_loss"].item():.4f}\t'
                      f'Cls: {loss_dict["cls_loss"].item():.4f}\t'
                      f'Acc: {current_acc:.2f}%')
                      
        except Exception as e:
            print(f"Error in VAE batch {batch_idx}: {e}")
            continue
    
    if total == 0:
        print("No successful batches in this VAE epoch!")
        return 0, 0
        
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    print(f'\nVAE Train Epoch {epoch} Summary:')
    print(f'  Total Loss: {epoch_loss:.6f}')
    print(f'  Reconstruction Loss: {running_recon_loss / len(dataloader):.6f}')
    print(f'  KL Loss: {running_kl_loss / len(dataloader):.6f}')
    print(f'  Classification Loss: {running_cls_loss / len(dataloader):.6f}')
    print(f'  Accuracy: {epoch_acc:.2f}% ({correct}/{total})')
    
    return epoch_loss, epoch_acc

def train_ed_epoch(model, device, dataloader, optimizer, epoch, criterion, mse_loss):
    model.train()
    running_loss = 0.0
    running_recon_loss = 0.0
    running_cls_loss = 0.0
    correct = 0
    total = 0
    
    print(f"\nStarting ED training epoch {epoch}")
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        try:
            # Forward pass
            classification_output, reconstructed = model(data)
            
            # Calculate losses
            loss_dict = model.get_loss(data, target, classification_output, reconstructed)
            total_loss = loss_dict['total_loss']
            
            # Check for NaN losses
            if torch.isnan(total_loss):
                print(f"NaN loss detected in batch {batch_idx}, skipping...")
                continue
                
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            running_loss += total_loss.item()
            running_recon_loss += loss_dict['recon_loss'].item()
            running_cls_loss += loss_dict['cls_loss'].item()
            
            _, predicted = torch.max(classification_output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 50 == 0:
                current_acc = 100. * correct / total if total > 0 else 0
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} '
                      f'({100. * batch_idx / len(dataloader):.0f}%)]\t'
                      f'Loss: {total_loss.item():.6f}\t'
                      f'Recon: {loss_dict["recon_loss"].item():.4f}\t'
                      f'Cls: {loss_dict["cls_loss"].item():.4f}\t'
                      f'Acc: {current_acc:.2f}%')
                      
        except Exception as e:
            print(f"Error in ED batch {batch_idx}: {e}")
            continue
    
    if total == 0:
        print("No successful batches in this ED epoch!")
        return 0, 0
        
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    print(f'\nED Train Epoch {epoch} Summary:')
    print(f'  Total Loss: {epoch_loss:.6f}')
    print(f'  Reconstruction Loss: {running_recon_loss / len(dataloader):.6f}')
    print(f'  Classification Loss: {running_cls_loss / len(dataloader):.6f}')
    print(f'  Accuracy: {epoch_acc:.2f}% ({correct}/{total})')
    
    return epoch_loss, epoch_acc

def validate_epoch(model, device, dataloader, epoch, criterion, mse_loss, model_type):
    model.eval()
    test_loss = 0
    test_recon_loss = 0
    test_cls_loss = 0
    test_kl_loss = 0
    correct = 0
    total = 0
    
    print(f"\nStarting {model_type.upper()} validation epoch {epoch}")
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            try:
                if model_type == "vae":
                    # Forward pass for VAE
                    classification_output, x_reconstructed, mu, logvar = model(data)
                    # Calculate losses
                    loss_dict = model.get_loss(data, target, classification_output, x_reconstructed, mu, logvar)
                    test_kl_loss += loss_dict['kl_loss'].item()
                else:
                    # Forward pass for ED
                    classification_output, reconstructed = model(data)
                    # Calculate losses
                    loss_dict = model.get_loss(data, target, classification_output, reconstructed)
                
                # Check for NaN
                if torch.isnan(loss_dict['total_loss']):
                    continue
                    
                test_loss += loss_dict['total_loss'].item()
                test_recon_loss += loss_dict['recon_loss'].item()
                test_cls_loss += loss_dict['cls_loss'].item()
                
                # Statistics
                _, predicted = torch.max(classification_output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue
    
    if total == 0:
        print("No successful validation batches!")
        return float('inf'), 0
        
    test_loss /= len(dataloader)
    test_recon_loss /= len(dataloader)
    test_cls_loss /= len(dataloader)
    test_acc = 100. * correct / total
    
    print(f'\n{model_type.upper()} Validation Epoch {epoch} Summary:')
    print(f'  Total Loss: {test_loss:.6f}')
    print(f'  Reconstruction Loss: {test_recon_loss:.6f}')
    if model_type == "vae":
        test_kl_loss /= len(dataloader)
        print(f'  KL Loss: {test_kl_loss:.6f}')
    print(f'  Classification Loss: {test_cls_loss:.6f}')
    print(f'  Accuracy: {test_acc:.2f}% ({correct}/{total})')
    
    return test_loss, test_acc

def train_model(dir_path, mod, num_epochs, pretrained_model_filename, test_model, batch_size):
    print("Loading data...")
    dataloaders, dataset_sizes = load_data(dir_path, batch_size)
    print("Done.")
    print(f"Dataset sizes: {dataset_sizes}")
    
    # Check first batch to see actual image size
    first_batch = next(iter(dataloaders["train"]))
    actual_img_size = first_batch[0].shape[2]  # Height
    print(f"Detected image size: {actual_img_size}x{actual_img_size}")
    
    # Initialize model
    if mod == "ed":
        model = GenConViTED(config)
        print(f"Using GenConViTED model (Final Version for {actual_img_size}x{actual_img_size} images)")
    else:
        model = GenConViTVAE(config)
        print(f"Using GenConViTVAE model (Final Version for {actual_img_size}x{actual_img_size} images)")
    
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
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Load pretrained if specified
    start_epoch = 0
    if pretrained_model_filename:
        model, optimizer, start_epoch, min_loss = load_pretrained(
            model, optimizer, pretrained_model_filename
        )
        print(f"Loaded pretrained model from epoch {start_epoch}")
    
    # Training loop
    torch.manual_seed(42)
    train_losses, train_accs, valid_losses, valid_accs = [], [], [], []
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    since = time.time()
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{start_epoch + num_epochs}")
        print(f"{'='*80}")
        
        # Training phase
        if mod == "vae":
            train_loss, train_acc = train_vae_epoch(
                model, device, dataloaders["train"], optimizer, epoch, criterion, mse_loss
            )
            valid_loss, valid_acc = validate_epoch(
                model, device, dataloaders["validation"], epoch, criterion, mse_loss, "vae"
            )
        else:
            train_loss, train_acc = train_ed_epoch(
                model, device, dataloaders["train"], optimizer, epoch, criterion, mse_loss
            )
            valid_loss, valid_acc = validate_epoch(
                model, device, dataloaders["validation"], epoch, criterion, mse_loss, "ed"
            )
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        
        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning rate: {current_lr:.8f}")
        
        # Save best model based on validation loss
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_val_acc = valid_acc
            print(f"🎉 New best validation loss: {best_val_loss:.6f} (Acc: {best_val_acc:.2f}%)")
            
            # Save best model immediately
            best_model_path = os.path.join("weight", f"best_genconvit_{mod}_epoch_{epoch+1}.pth")
            os.makedirs("weight", exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "config": config
            }, best_model_path)
            print(f"Best model saved to: {best_model_path}")
        
        print(f"Best validation loss so far: {best_val_loss:.6f} (Acc: {best_val_acc:.2f}%)")
    
    time_elapsed = time.time() - since
    print(f"\n{'='*80}")
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"{'='*80}")
    
    # Save final model
    print("\nSaving final model...")
    file_path = os.path.join(
        "weight",
        f'genconvit_{mod}_final_{time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime())}',
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
        "final_loss": valid_loss,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "config": config
    }
    
    weight_path = f"{file_path}.pth"
    torch.save(state, weight_path)
    print(f"Final model saved to: {weight_path}")
    print(f"Training history saved to: {file_path}.pkl")
    
    if test_model:
        test(model, dataloaders, dataset_sizes, mod, weight_path)

def test(model, dataloaders, dataset_sizes, mod, weight_path):
    print(f"\n{'='*80}")
    print("Running test...")
    print(f"{'='*80}")
    
    model.eval()
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloaders["test"]):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            try:
                if mod == "ed":
                    outputs, _ = model(inputs)
                else:
                    outputs, _, _, _ = model(inputs)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Per-class accuracy
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_correct[label] = class_correct.get(label, 0) + (predicted[i] == labels[i]).item()
                    class_total[label] = class_total.get(label, 0) + 1
                
                if batch_idx % 20 == 0:
                    current_acc = 100 * correct / total if total > 0 else 0
                    print(f'Test Progress: {batch_idx}/{len(dataloaders["test"])} '
                          f'({100. * batch_idx / len(dataloaders["test"]):.1f}%) '
                          f'- Current Acc: {current_acc:.2f}%')
                          
            except Exception as e:
                print(f"Error in test batch {batch_idx}: {e}")
                continue
    
    accuracy = 100 * correct / total
    print(f'\n🎯 Final Test Results:')
    print(f'   Correct: {correct}/{total}')
    print(f'   Overall Accuracy: {accuracy:.2f}%')
    
    # Per-class results
    print(f'\n📊 Per-class Results:')
    for class_id in sorted(class_correct.keys()):
        if class_total[class_id] > 0:
            class_acc = 100 * class_correct[class_id] / class_total[class_id]
            class_name = "Real" if class_id == 0 else "Fake"
            print(f'   Class {class_id} ({class_name}): {class_correct[class_id]}/{class_total[class_id]} ({class_acc:.2f}%)')
    
    print(f"{'='*80}")

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
    print("🚀 GenConViT Complete Training Script")
    print("=" * 80)
    
    start_time = perf_counter()
    path, mod, epoch, pretrained_model_filename, test_model, batch_size = gen_parser()
    
    print(f"Configuration:")
    print(f"  📁 Data path: {path}")
    print(f"  🤖 Model: {mod.upper()}")
    print(f"  🔄 Epochs: {epoch}")
    print(f"  📦 Batch size: {batch_size}")
    print(f"  💻 Device: {device}")
    print(f"  🔄 Pretrained: {pretrained_model_filename}")
    print(f"  🧪 Test after training: {test_model}")
    print("=" * 80)
    
    try:
        train_model(path, mod, epoch, pretrained_model_filename, test_model, batch_size)
        print("\n✅ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    end_time = perf_counter()
    print(f"\n⏱️ Total execution time: {end_time - start_time:.2f} seconds")
    print("🎉 All done!")
    return 0

if __name__ == "__main__":
    sys.exit(main())