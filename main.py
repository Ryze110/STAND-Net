import os
import torch
import torch.optim as optim
from loaddata import load_and_normalize_dataset, prepare_data_loaders
from train import train_model
from evaluate_plot import evaluate_model, print_metrics, save_metrics, plot_training_history, plot_denoised_samples
from loss import SimpleMSELoss
from units import device    
import numpy as np
from createmodel import create_model
import datetime
import json


def main():
    """Main function to train the SNN Attention model"""
    
    # Fixed configuration parameters
    config = {
        'model': 'staanet',
        'epochs': 1,
        'batch_size': 32,
        'lr': 0.001,
        'patience': 10,
        'data_dir': 'contaminated_eeg_datasets',
        'artifact_type': 'ECG',  # Modify as needed
        'output_dir': 'staanet_results',
        'T': 8,
        'loss_type': 'mse',
    }
    
    # Get the current time, formatted as YYYYMMDD_HHMMSS
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create output directory
    output_dir = f'{config["output_dir"]}/{config["artifact_type"]}_{current_time}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset using EEGDnet-style normalization
    print("Loading dataset using normalization...")
    X_train, y_train, X_val, y_val, X_test, y_test, snrs_test = load_and_normalize_dataset(
        config['data_dir'], config['artifact_type'], use_data_norm=True
    )
    
    if X_train is None:
        print("Failed to load dataset, exiting program")
        return
    
    # Prepare standard data loaders
    train_loader, val_loader, test_loader = prepare_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, snrs_test, 
        batch_size=config['batch_size']
    )
    
    # Create SNN Attention model
    input_size = X_train.shape[1] if len(X_train.shape) == 2 else X_train.shape[2]
    model = create_model(config['model'], input_size)
    model = model.to(device)
    
    # Print model structure
    print(f"\n{config['model'].upper()} Model Structure:")
    print(model)
    
    # Use simple MSE loss function
    criterion = SimpleMSELoss()
    print("Using MSE loss function")
    
    # Regular training process
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    print(f"\nStarting training for {config['model'].upper()} model...")
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=config['epochs'], early_stopping_patience=config['patience'], 
        device=device, T=config['T']
    )
    
    # Save the final model
    model_filename = f'{config["model"]}_final.pth'
    torch.save(model.state_dict(), os.path.join(output_dir, model_filename))
    
    # Save training loss history to CSV file
    loss_filename = f'{config["model"]}_loss_history.csv'
    loss_filepath = os.path.join(output_dir, loss_filename)
    
    with open(loss_filepath, 'w') as f:
        f.write('epoch,train_loss,val_loss\n')
        for i in range(len(history['train_loss'])):
            f.write(f"{i+1},{history['train_loss'][i]:.6f},{history['val_loss'][i]:.6f}\n")
    
    print(f"Training loss history saved to {loss_filepath}")
    
    # Evaluate the model
    print("\nEvaluating model performance...")
    metrics, predictions = evaluate_model(
        model, test_loader, device=device, T=config['T']
    )
    
    # Print and save metrics
    print_metrics(metrics)
    metrics_filename = f'{config["model"]}_metrics.json'
    save_metrics(metrics, f"{config['model'].upper()} Model", 
                os.path.join(output_dir, metrics_filename))
    
    # Visualize results
    plot_training_history(history, 
                        save_path=os.path.join(output_dir, f'{config["model"]}_training_history.png'))
    plot_denoised_samples(predictions, 
                        save_path=os.path.join(output_dir, f'{config["model"]}_denoising_results.png'))
    
    
    print(f"\nTraining completed! Results saved in: {output_dir}")
    
    # Save configuration information
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    main()