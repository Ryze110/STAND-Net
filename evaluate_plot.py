import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import signal
import time
import json
from tqdm import tqdm
from spikingjelly.clock_driven import functional
from units import device 

def evaluate_model(model, test_loader, device=device, T=8):
    """
    Evaluate model performance on test dataset
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        T: Time steps for SNN models
        
    Returns:
        metrics: Dictionary of evaluation metrics
        predictions: Sample predictions for visualization
        snr_performance: Performance metrics by SNR level
    """
    # Check model type
    is_snn_model = any(cls_name in model.__class__.__name__ for cls_name in ['SNNAutoencoder'])
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize metric collectors
    all_mse = []
    all_mae = []
    all_rrmse = []
    all_snr_before = []
    all_snr_after = []
    all_psnr_before = []
    all_psnr_after = []
    all_corr = []
    snr_groups = {}
    has_snr_info = False
    
    # Save samples for visualization
    sample_noisy = None
    sample_clean = None
    sample_denoised = None
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, desc='Evaluating')):
            # Check if SNR information is available
            if len(data) == 3:
                noisy, clean, snrs = data
                has_snr_info = True
            else:
                noisy, clean = data
                snrs = None
                        
            noisy, clean = noisy.to(device), clean.to(device)
            if has_snr_info:
                snrs = snrs.to(device)
            
            # Reset neural states for SNN models
            if is_snn_model:
                functional.reset_net(model)
            
            # Forward propagation with T parameter for SNN models
            if is_snn_model:
                denoised = model(noisy, T=T)
            else:
                denoised = model(noisy)
            
            # Save samples for visualization (only for the first batch)
            if i == 0:
                sample_noisy = noisy[:5].cpu().numpy().copy()
                sample_clean = clean[:5].cpu().numpy().copy()
                sample_denoised = denoised[:5].cpu().numpy().copy()
                
            # Compute batch metrics - keep on GPU as much as possible
            # MSE
            mse = torch.mean((denoised - clean) ** 2, dim=-1)
            
            # MAE
            mae = torch.mean(torch.abs(denoised - clean), dim=-1)
            
            # RRMSE - computed based on formula
            energy_batch = torch.mean(clean ** 2, dim=-1) + 1e-8
            rrmse_batch = torch.sqrt(mse / energy_batch)
            
            # Transfer to CPU
            mse_cpu = mse.cpu().numpy()
            mae_cpu = mae.cpu().numpy()
            rrmse_cpu = rrmse_batch.cpu().numpy()
            noisy_cpu = noisy.cpu().numpy()
            clean_cpu = clean.cpu().numpy()
            denoised_cpu = denoised.cpu().numpy()
            
            # Compute batch SNR, PSNR, and other metrics
            batch_snr_before = []
            batch_snr_after = []
            batch_psnr_before = []
            batch_psnr_after = []
            batch_corr = []
            
            for j in range(noisy.size(0)):
                # Compute SNR
                noise_before = noisy_cpu[j] - clean_cpu[j]
                noise_power_before = np.mean(noise_before ** 2)
                signal_power = np.mean(clean_cpu[j] ** 2)
                
                if noise_power_before > 0:
                    snr_before = 10 * np.log10(signal_power / noise_power_before)
                else:
                    snr_before = 100  # High value for no noise
                
                noise_after = denoised_cpu[j] - clean_cpu[j]
                noise_power_after = np.mean(noise_after ** 2)
                
                if noise_power_after > 0:
                    snr_after = 10 * np.log10(signal_power / noise_power_after)
                else:
                    snr_after = 100
                
                batch_snr_before.append(snr_before)
                batch_snr_after.append(snr_after)
                
                # Compute PSNR
                max_signal = np.max(np.abs(clean_cpu[j]))
                
                if noise_power_before > 0:
                    psnr_before = 20 * np.log10(max_signal / np.sqrt(noise_power_before))
                else:
                    psnr_before = 100
                
                if noise_power_after > 0:
                    psnr_after = 20 * np.log10(max_signal / np.sqrt(noise_power_after))
                else:
                    psnr_after = 100
                
                batch_psnr_before.append(psnr_before)
                batch_psnr_after.append(psnr_after)
                
                # Compute correlation coefficient
                corr = np.corrcoef(clean_cpu[j].flatten(), denoised_cpu[j].flatten())[0, 1]
                batch_corr.append(corr)
                
                          
            # Add batch average metrics
            all_mse.extend(mse_cpu)
            all_mae.extend(mae_cpu)
            all_rrmse.extend(rrmse_cpu)
            all_snr_before.extend(batch_snr_before)
            all_snr_after.extend(batch_snr_after)
            all_psnr_before.extend(batch_psnr_before)
            all_psnr_after.extend(batch_psnr_after)
            all_corr.extend(batch_corr)
    
    # Compute average metrics
    avg_mse = np.mean(all_mse)
    avg_mae = np.mean(all_mae)
    avg_rrmse = np.mean(all_rrmse)
    avg_snr_before = np.mean(all_snr_before)
    avg_snr_after = np.mean(all_snr_after)
    avg_psnr_before = np.mean(all_psnr_before)
    avg_psnr_after = np.mean(all_psnr_after)
    avg_corr = np.mean(all_corr)
    
    # Compile metrics
    metrics = {
        'mse': float(avg_mse),
        'mae': float(avg_mae),
        'rrmse': float(avg_rrmse),
        'snr_before': float(avg_snr_before),
        'snr_after': float(avg_snr_after),
        'snr_improvement': float(avg_snr_after - avg_snr_before),
        'psnr_before': float(avg_psnr_before),
        'psnr_after': float(avg_psnr_after),
        'psnr_improvement': float(avg_psnr_after - avg_psnr_before),
        'correlation': float(avg_corr)
    }
    
    # Prepare sample predictions for visualization
    predictions = {
        'noisy': sample_noisy,
        'clean': sample_clean,
        'denoised': sample_denoised
    }
    
    return metrics, predictions

# --------------------------------
# Visualization Functions
# --------------------------------

def plot_training_history(history, save_path='training_history.png'):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_denoised_samples(predictions, save_path='denoising_results.png'):
    """Plot examples of noisy, clean, and denoised EEG signals"""
    n_samples = len(predictions['noisy'])
    
    plt.figure(figsize=(15, 3*n_samples))
    
    for i in range(n_samples):
        # Time domain
        plt.subplot(n_samples, 2, 2*i+1)
        plt.plot(predictions['noisy'][i, 0], 'gray', alpha=0.7, label='Noisy')
        plt.plot(predictions['clean'][i, 0], 'g', label='Clean')
        plt.plot(predictions['denoised'][i, 0], 'b', label='Denoised')
        plt.title(f'Sample {i+1}: Time Domain')
        plt.xlabel('Time Points')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Frequency domain (PSD)
        plt.subplot(n_samples, 2, 2*i+2)
        fs = 250  # Assuming 250 Hz sampling rate for EEG
        
        f, psd_noisy = signal.welch(predictions['noisy'][i, 0], fs, nperseg=256)
        f, psd_clean = signal.welch(predictions['clean'][i, 0], fs, nperseg=256)
        f, psd_denoised = signal.welch(predictions['denoised'][i, 0], fs, nperseg=256)
        
        plt.semilogy(f, psd_noisy, 'gray', alpha=0.7, label='Noisy')
        plt.semilogy(f, psd_clean, 'g', label='Clean')
        plt.semilogy(f, psd_denoised, 'b', label='Denoised')
        plt.title(f'Sample {i+1}: Power Spectral Density')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD (μV²/Hz)')
        plt.legend()
        plt.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_metrics(metrics, model_name, save_path='model_metrics.json'):
    """Save evaluation metrics to a JSON file"""
    metrics['model'] = model_name
    metrics['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to {save_path}")

def print_metrics(metrics):
    """Print evaluation metrics in a formatted way"""
    print("\n" + "="*50)
    print("Model Evaluation Metrics")
    print("="*50)
    
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"RRMSE: {metrics['rrmse']:.6f}")
    print(f"SNR Before Denoising: {metrics['snr_before']:.2f} dB")
    print(f"SNR After Denoising: {metrics['snr_after']:.2f} dB")
    print(f"SNR Improvement: {metrics['snr_improvement']:.2f} dB")
    print(f"PSNR Before Denoising: {metrics['psnr_before']:.2f} dB")
    print(f"PSNR After Denoising: {metrics['psnr_after']:.2f} dB")
    print(f"PSNR Improvement: {metrics['psnr_improvement']:.2f} dB")
    print(f"Correlation with Clean Signal: {metrics['correlation']:.4f}")
    print("="*50)