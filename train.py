import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from spikingjelly.clock_driven import functional
from units import device 

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=50, early_stopping_patience=10, device=device, T=8):
    """
    Train the SNN Attention model
    
    Args:
        model: SNN Attention model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function (fixed as MSE)
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        early_stopping_patience: Early stopping patience
        device: Computing device
        T: Simulation time steps for the SNN model
    
    Returns:
        model: Trained model
        history: Dictionary containing training history
    """
    # Initialize early stopping and history tracking variables
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # Create a scaler for mixed precision training
    scaler = GradScaler()
    
    # Start training loop
    for epoch in range(num_epochs):
        # ==== Training phase === =
        model.train()
        running_loss = 0.0
        
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for noisy, clean in train_progress:
            noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
            
            # Forward pass
            with autocast():
                # Reset SNN neuron states
                functional.reset_net(model)
                
                # Forward propagation through the SNN model
                outputs = model(noisy, T=T)
                
                # Compute MSE loss
                loss = criterion(outputs, clean)
            
            # Backward pass and optimization
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Accumulate loss
            running_loss += loss.item() * noisy.size(0)
            train_progress.set_postfix({'loss': f'{loss.item():.6f}'})
            
        # Compute average loss for the entire training set
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # ==== Validation phase === =
        model.eval()
        running_val_loss = 0.0
        
        val_progress = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for noisy, clean in val_progress:
                noisy, clean = noisy.to(device), clean.to(device)
                
                # Reset SNN neuron states
                functional.reset_net(model)
                
                # Forward propagation through the SNN model
                outputs = model(noisy, T=T)
                
                # Compute validation loss
                loss = criterion(outputs, clean)
                
                # Accumulate validation loss
                running_val_loss += loss.item() * noisy.size(0)
                val_progress.set_postfix({'loss': f'{loss.item():.6f}'})
                
        # Compute average loss for the entire validation set
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # Update learning rate scheduler
        scheduler.step(epoch_val_loss)
        
        # Print progress information
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {epoch_train_loss:.6f}, '
              f'Val Loss: {epoch_val_loss:.6f}')
        
        # ==== Early stopping check === =
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Stop training if early stopping condition is met
        if patience_counter >= early_stopping_patience:
            print(f'Early stopping triggered after epoch {epoch+1}')
            break
    
    # Load the best model weights based on validation performance
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Create training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses
    }
    
    # Print GPU memory usage
    if torch.cuda.is_available():
        print(f"Allocated GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Reserved GPU memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    return model, history