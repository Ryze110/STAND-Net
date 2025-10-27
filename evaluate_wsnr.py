import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import os
import argparse
import json
from tqdm import tqdm
import pywt  # For wavelet decomposition
import signal

# Import model creation and evaluation functions from STC_AutoEncoderNet
from createmodel import create_model
from loaddata import load_dataset, prepare_data_loaders, load_and_normalize_dataset
from units import device
from evaluate_plot import print_metrics, save_metrics, plot_denoised_samples
from spikingjelly.clock_driven import functional

def load_trained_model(model_path, model_type, input_size, device='cuda'):
    """
    Load a trained model.

    Args:
        model_path (str): Path to the model weights file (.pth).
        model_type (str): Type of the model, e.g., 'fcnn', 'cnn', 'snn_autoencoder'.
        input_size (int): Size of the input data.
        device (str): Device to use ('cuda' or 'cpu').
    """
    # Create a model instance
    model = create_model(model_type, input_size)

    # Load model weights
    state_dict = torch.load(model_path)

    # Handle key names in the weight dictionary (for models saved with multi-GPU training)
    if list(state_dict.keys())[0].startswith('module.'):
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        state_dict = new_state_dict

    # Load weights into the model
    model.load_state_dict(state_dict)

    # Special handling for the DeT model, convert it to double type
    if 'DeT' in model.__class__.__name__:
        model = model.double()
        print("DeT model has been converted to double precision.")

    # Move the model to the specified device and set it to evaluation mode
    model = model.to(device)
    model.eval()

    return model

def calculate_relative_wavelet_energy(coeffs):
    """Calculate the relative energy of wavelet sub-bands."""
    # Ensure the coefficient format is correct
    if not isinstance(coeffs, list):
        raise ValueError("Wavelet coefficients must be in list format.")

    # Calculate the total energy of all coefficients
    total_energy = 0
    for i, coeff in enumerate(coeffs):
        # Ensure each coefficient is an array
        if not isinstance(coeff, np.ndarray):
            raise ValueError(f"The {i}-th coefficient is not a numpy array, but {type(coeff)}")

        total_energy += np.sum(np.abs(coeff) ** 2)

    total_energy = max(total_energy, 1e-10)  # Avoid division by zero

    # Calculate the relative energy of each sub-band
    rel_energies = []
    for coeff in coeffs:
        energy = np.sum(np.abs(coeff) ** 2)
        rel_energies.append(energy / total_energy)

    return np.array(rel_energies)

def calculate_relative_wavelet_entropy(rel_energies):
    """Calculate the relative entropy of wavelet sub-bands."""
    # Avoid log(0)
    rel_energies = np.clip(rel_energies, 1e-10, 1.0)
    return -rel_energies * np.log(rel_energies)

def calculate_wsnr(clean_signal, denoised_signal, wavelet='db9', level=5):
    """
    Calculate the Wavelet Signal-to-Noise Ratio (WSNR).

    Args:
        clean_signal: The clean signal.
        denoised_signal: The denoised signal.
        wavelet: The wavelet basis function.
        level: The decomposition level (will be automatically adjusted based on signal length).

    Returns:
        wsnr_e: Energy-based Wavelet Signal-to-Noise Ratio.
        wsnr_h: Entropy-based Wavelet Signal-to-Noise Ratio.
    """
    # Ensure correct signal dimensions
    clean_signal = clean_signal.ravel()
    denoised_signal = denoised_signal.ravel()

    # Automatically adjust the maximum decomposition level based on signal length
    signal_length = len(clean_signal)

    # Calculate the theoretically possible maximum level (considering wavelet filter length)
    wavelet_obj = pywt.Wavelet(wavelet)
    filter_length = wavelet_obj.dec_len

    # Calculate the maximum possible level according to the formula to avoid boundary effects
    # pywt suggests max_level = log2(signal_length / (filter_length - 1))
    max_level = pywt.dwt_max_level(signal_length, filter_length)

    # Use a smaller level to avoid warnings
    actual_level = min(level, max_level)
    if actual_level < level:
        # print(f"Automatically adjusting wavelet decomposition level: {level} -> {actual_level} (signal length: {signal_length})")
        pass

    try:
        # Perform wavelet decomposition
        clean_coeffs = pywt.wavedec(clean_signal, wavelet, level=actual_level)
        denoised_coeffs = pywt.wavedec(denoised_signal, wavelet, level=actual_level)

        # Calculate relative energy
        clean_rel_energies = calculate_relative_wavelet_energy(clean_coeffs)

        # Calculate relative entropy
        clean_rel_entropies = calculate_relative_wavelet_entropy(clean_rel_energies)

        # Calculate WSNR for each sub-band
        wsnr_e_sum = 0
        wsnr_h_sum = 0

        for j in range(len(clean_coeffs)):
            # Extract sub-band signals
            y_clean = clean_coeffs[j]
            y_denoised = denoised_coeffs[j]

            # Ensure array dimensions match
            if y_clean.shape != y_denoised.shape:
                # Adjust if dimensions do not match
                min_len = min(len(y_clean), len(y_denoised))
                y_clean = y_clean[:min_len]
                y_denoised = y_denoised[:min_len]

            # Calculate signal energy and error energy
            clean_energy = np.sum(np.abs(y_clean) ** 2)
            error_energy = np.sum(np.abs(y_clean - y_denoised) ** 2)

            # Avoid division by zero
            if error_energy < 1e-10:
                error_energy = 1e-10

            if clean_energy < 1e-10:
                continue  # Skip sub-bands with near-zero energy

            # Calculate the WSNR contribution of this sub-band
            snr_j = 10 * np.log10(clean_energy / error_energy)

            # Calculate weighted WSNR based on energy and entropy weights
            wsnr_e_j = clean_rel_energies[j] * snr_j
            wsnr_h_j = clean_rel_entropies[j] * snr_j

            wsnr_e_sum += wsnr_e_j
            wsnr_h_sum += wsnr_h_j

        return wsnr_e_sum, wsnr_h_sum

    except Exception as e:
        # Capture detailed error
        import traceback
        print(f"Error calculating WSNR: {e}")
        print(traceback.format_exc())
        return 0, 0

def evaluate_trained_model(model, test_loader, device='cuda', T=8, num_samples=5):
    """
    Evaluate the performance of a trained model.

    Args:
        model: The loaded model.
        test_loader: The test data loader.
        device: The device to use ('cuda' or 'cpu').
        T: The number of time steps for SNN models.
        num_samples: The number of samples to save for visualization.
    """
    # Check model type
    is_snn_model = any(cls_name in model.__class__.__name__ for cls_name in ['SNN_FCNN', 'SNNAutoencoder'])
    is_det_model = 'DeT' in model.__class__.__name__

    # Set the model to evaluation mode
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
    # Add WSNR metric collectors
    all_wsnr_e = []
    all_wsnr_h = []
    has_snr_info = False

    # Save samples for visualization
    sample_noisy = None
    sample_clean = None
    sample_denoised = None

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, desc='Evaluating model')):
            # Check if there is SNR information
            if len(data) == 3:
                noisy, clean, snrs = data
                has_snr_info = True
            else:
                noisy, clean = data
                snrs = None

            # For the DeT model, ensure the input data is of double type
            if is_det_model and noisy.dtype != torch.float64:
                noisy = noisy.double()
                clean = clean.double()
                if snrs is not None:
                    snrs = snrs.double()

            noisy, clean = noisy.to(device), clean.to(device)

            # Reset neural states for SNN models
            if is_snn_model:
                functional.reset_net(model)

            # Forward propagation with T parameter for SNN models
            if is_snn_model:
                denoised = model(noisy, T=T)
            else:
                denoised = model(noisy)

            # Save samples for visualization only for the first batch
            if i == 0:
                # Ensure not to exceed the batch size
                samples_to_save = min(num_samples, noisy.size(0))
                sample_noisy = noisy[:samples_to_save].cpu().numpy().copy()
                sample_clean = clean[:samples_to_save].cpu().numpy().copy()
                sample_denoised = denoised[:samples_to_save].cpu().numpy().copy()

            # Calculate batch metrics
            # MSE
            mse = torch.mean((denoised - clean) ** 2, dim=-1)

            # MAE
            mae = torch.mean(torch.abs(denoised - clean), dim=-1)

            # RRMSE - calculated according to the formula
            energy_batch = torch.mean(clean ** 2, dim=-1) + 1e-8
            rrmse_batch = torch.sqrt(mse / energy_batch)

            # Transfer to CPU
            mse_cpu = mse.cpu().numpy()
            mae_cpu = mae.cpu().numpy()
            rrmse_cpu = rrmse_batch.cpu().numpy()
            noisy_cpu = noisy.cpu().numpy()
            clean_cpu = clean.cpu().numpy()
            denoised_cpu = denoised.cpu().numpy()

            # Batch calculate SNR, PSNR, and other metrics
            batch_snr_before = []
            batch_snr_after = []
            batch_psnr_before = []
            batch_psnr_after = []
            batch_corr = []
            # Batch calculate WSNR metrics
            batch_wsnr_e = []
            batch_wsnr_h = []

            for j in range(noisy.size(0)):
                # Calculate SNR
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

                # Calculate PSNR
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

                # Calculate correlation coefficient
                corr = np.corrcoef(clean_cpu[j].flatten(), denoised_cpu[j].flatten())[0, 1]
                batch_corr.append(corr)

                # Calculate WSNR metrics
                try:
                    # Ensure input is a 1D array
                    clean_flat = clean_cpu[j].flatten()
                    denoised_flat = denoised_cpu[j].flatten()

                    # Check array length
                    if len(clean_flat) >= 32:  # Ensure sufficient length for wavelet decomposition
                        wsnr_e, wsnr_h = calculate_wsnr(
                            clean_flat,
                            denoised_flat,
                            wavelet='db9',  # Use Daubechies9 wavelet
                            level=5  # 5-level decomposition
                        )
                        batch_wsnr_e.append(wsnr_e)
                        batch_wsnr_h.append(wsnr_h)
                    else:
                        print(f"Signal too short ({len(clean_flat)} points), skipping WSNR calculation")
                        batch_wsnr_e.append(0)
                        batch_wsnr_h.append(0)
                except Exception as e:
                    print(f"Error during WSNR calculation: {e}")
                    print(f"Signal shape: clean={clean_cpu[j].shape}, denoised={denoised_cpu[j].shape}")
                    batch_wsnr_e.append(0)
                    batch_wsnr_h.append(0)

            # Add batch average metrics
            all_mse.extend(mse_cpu)
            all_mae.extend(mae_cpu)
            all_rrmse.extend(rrmse_cpu)
            all_snr_before.extend(batch_snr_before)
            all_snr_after.extend(batch_snr_after)
            all_psnr_before.extend(batch_psnr_before)
            all_psnr_after.extend(batch_psnr_after)
            all_corr.extend(batch_corr)
            # Add WSNR metrics
            all_wsnr_e.extend(batch_wsnr_e)
            all_wsnr_h.extend(batch_wsnr_h)

    # Calculate average metrics
    avg_mse = np.mean(all_mse)
    avg_mae = np.mean(all_mae)
    avg_rrmse = np.mean(all_rrmse)
    avg_snr_before = np.mean(all_snr_before)
    avg_snr_after = np.mean(all_snr_after)
    avg_psnr_before = np.mean(all_psnr_before)
    avg_psnr_after = np.mean(all_psnr_after)
    avg_corr = np.mean(all_corr)
    # Calculate WSNR average
    avg_wsnr_e = np.mean(all_wsnr_e)
    avg_wsnr_h = np.mean(all_wsnr_h)

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
        'correlation': float(avg_corr),
        # Add WSNR metrics
        'wsnr_e': float(avg_wsnr_e),
        'wsnr_h': float(avg_wsnr_h)
    }

    # Prepare sample predictions for visualization
    predictions = {
        'noisy': sample_noisy,
        'clean': sample_clean,
        'denoised': sample_denoised
    }

    return metrics, predictions, denoised_cpu

# Add a new function to plot individual samples and save them separately
def plot_individual_samples(predictions, output_dir, fs=256):
    """
    Plot and save images for each sample separately.

    Parameters:
    predictions: Dictionary containing noisy, clean, and denoised signals.
    output_dir: Output directory.
    fs: Signal sampling rate, default is 256Hz.
    """
    n_samples = len(predictions['noisy'])
    
    for i in range(n_samples):
        plt.rcParams['font.family'] = 'Arial'  # Font family, e.g., 'serif', 'sans-serif', 'monospace'
        plt.figure(figsize=(15, 5))
        
        # Time-domain signal
        plt.subplot(1, 1, 1)
        plt.plot(predictions['noisy'][i, 0], 'gray', linewidth=2, alpha=0.7, label='Noise Signal')
        plt.plot(predictions['clean'][i, 0], 'g', linewidth=2, label='Clean Signal')
        plt.plot(predictions['denoised'][i, 0], 'b', linewidth=2, label='Denoised Signal')
        plt.title(f'sample {i+1} ',size=15)
        plt.tick_params(axis='both', labelsize=15)  # 'both' means setting both x and y axes
        plt.xlabel('Points',size=15)
        plt.ylabel('Amplitude(mV)',size=15)
        plt.legend(fontsize=15,loc='upper left')
        plt.grid(True, alpha=0.3)       
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_{i+1}_denoising_results.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved visualization results for {n_samples} samples individually to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate EEG signal denoising model')
    
    # Add command line arguments
    parser.add_argument('--model_path', type=str, default='E:\PycharmProjects\STC_AutoEncoderNet_EEG/benchmark_model_results_EOG\snn_attention_norm-eegdnet_loss-mse_EOG_20250421_093038\snn_attention_norm-eegdnet_final.pth',
                        help='Path to the model file')
    parser.add_argument('--model_type', type=str, default='snn_attention',
                        choices=['tcn', 'fcnn', 'cnn', 'complex_cnn', 'lstm', 'novel_cnn', 
                                 'snn_autoencoder', 'ct_dcenet','1d_resnet_cnn','duocl','det','snn_attention'],
                        help='Type of the model')
    parser.add_argument('--data_dir', type=str, default='E:/PycharmProjects/STC_AutoEncoderNet_EEG/contaminated_eeg_datasets_0408',
                        help='Base directory of the dataset')
    parser.add_argument('--artifact_type', type=str, default='EOG+ECG',
                        choices=['EMG', 'EOG', 'ECG', 'EMG+EOG', 'EMG+ECG', 'EOG+ECG', 'EMG+EOG+ECG'],
                        help='Type of artifact')
    parser.add_argument('--output_dir', type=str, default='model_evaluation_results_0421',
                        help='Output directory')
    parser.add_argument('--normalize', type=str, default='eegdnet',
                        choices=['none', 'eegdnet'],
                        help='Which normalization method to use')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size')
    parser.add_argument('--T', type=int, default=8,
                        help='Time steps for SNN model')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of samples for visualization')
    parser.add_argument('--individual_plots', action='store_true',default='true',
                        help='Whether to save visualization results for each sample individually')
    
    args = parser.parse_args()
    
    # Check if the model file exists
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        return
    
    # Create output directory
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    model_output_dir = os.path.join(args.output_dir, model_name, args.artifact_type)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Load dataset
    if args.normalize == 'eegdnet':
        print("Using EEGDnet-style normalization...")
        X_train, y_train, X_val, y_val, X_test, y_test, snrs_test = load_and_normalize_dataset(
            args.data_dir, args.artifact_type, use_eegdnet_norm=True
        )
    else:
        print("Not using special normalization...")
        X_train, y_train, X_val, y_val, X_test, y_test, snrs_test = load_dataset(
            args.data_dir, args.artifact_type
        )
    
    if X_test is None:
        print("Failed to load dataset, exiting.")
        return
    
    # Prepare test data loader
    train_loader, val_loader, test_loader = prepare_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, snrs_test, batch_size=args.batch_size
    )
    
    # Calculate input size
    input_size = X_test.shape[1] if len(X_test.shape) == 2 else X_test.shape[2]
    
    print(f"\nStarting evaluation for model: {model_name}")
    print(f"Model type: {args.model_type}")
    
    try:
        # Load model
        model = load_trained_model(args.model_path, args.model_type, input_size, device)
        
        # Evaluate model
        metrics, predictions, denoised_signals = evaluate_trained_model(
            model, test_loader, device, T=args.T, num_samples=args.num_samples
        )
        
        # Print evaluation metrics
        print_metrics(metrics)
        
        # Save metrics
        metrics_filename = f'{model_name}_metrics.json'
        save_metrics(metrics, model_name, os.path.join(model_output_dir, metrics_filename))
        
        # Plot summary results
        # plot_denoised_samples(predictions, save_path=os.path.join(model_output_dir, f'{model_name}_denoising_results.png'))
        
        # If needed, plot and save individual plots for each sample
        if args.individual_plots:
            plot_individual_samples(predictions, model_output_dir)
        
        # Save denoised signals
        np.save(os.path.join(model_output_dir, f'{model_name}_denoised_signals.npy'), denoised_signals)
        
        print(f"Model {model_name} evaluation complete, results saved to {model_output_dir}")
        
    except Exception as e:
        print(f"Error during model evaluation: {e}")

if __name__ == "__main__":
    main()