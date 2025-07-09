import os
import numpy as np
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split

def load_data():
    """
    Load clean EEG and various artifact data
    
    Returns:
        eeg_data: Clean EEG data
        artifacts: Dictionary containing various artifacts
    """
    print("Loading data...")
    
    # Load clean EEG data
    eeg_data = np.load('data\EEGdenoiseNet\EEG_all_epochs.npy')
    
    # Load various artifact data
    emg_data = np.load('data\EEGdenoiseNet\EMG_all_epochs.npy')
    eog_data = np.load('data\EEGdenoiseNet\EOG_all_epochs.npy')
    ecg_data = np.load('data\mit-bih-arrhythmia-database-1.0.0\ecg_artifacts.npy')
    
    artifacts = {
        'EMG': emg_data,
        'EOG': eog_data,
        'ECG': ecg_data
    }
    
    print(f"Data loaded successfully:")
    print(f"- Clean EEG data: {eeg_data.shape}")
    print(f"- EMG artifact data: {emg_data.shape}")
    print(f"- EOG artifact data: {eog_data.shape}")
    print(f"- ECG artifact data: {ecg_data.shape}")
    
    return eeg_data, artifacts

def calculate_snr(clean_signal, noise_signal):
    """
    Calculate Signal-to-Noise Ratio (SNR)
    
    Args:
        clean_signal: Clean signal
        noise_signal: Noise signal
    
    Returns:
        snr: Signal-to-Noise Ratio (dB)
    """
    clean_power = np.mean(clean_signal ** 2)
    noise_power = np.mean(noise_signal ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(clean_power / noise_power)
    return snr

def adjust_artifact_for_snr(clean_eeg, artifact, target_snr):
    """
    Adjust artifact amplitude to achieve target SNR
    
    Args:
        clean_eeg: Clean EEG signal
        artifact: Artifact signal
        target_snr: Target SNR (dB)
    
    Returns:
        scaled_artifact: Adjusted artifact
        alpha: Scaling factor
    """
    # Ensure shape matches
    if clean_eeg.shape != artifact.shape:
        # If artifact is multi-channel, take the first channel
        if len(artifact.shape) > 1 and artifact.shape[1] > 1:
            artifact = artifact[:, 0]
        
        # Adjust length
        min_length = min(len(clean_eeg), len(artifact))
        clean_eeg = clean_eeg[:min_length]
        artifact = artifact[:min_length]
    
    # Calculate current artifact power
    clean_power = np.mean(clean_eeg ** 2)
    artifact_power = np.mean(artifact ** 2)
    
    # Calculate target noise power
    target_noise_power = clean_power / (10 ** (target_snr / 10))
    
    # Calculate scaling factor
    if artifact_power == 0:
        alpha = 0
    else:
        alpha = np.sqrt(target_noise_power / artifact_power)
    
    # Scale artifact
    scaled_artifact = alpha * artifact
    
    return scaled_artifact, alpha

def generate_contaminated_eeg(clean_eeg, artifacts, artifact_types, target_snr=0):
    """
    Generate EEG signal contaminated with artifacts
    
    Args:
        clean_eeg: Clean EEG signal
        artifacts: Dictionary of artifacts
        artifact_types: List of artifact types to add
        target_snr: Target SNR (dB)
    
    Returns:
        contaminated_eeg: Contaminated EEG signal
        alphas: Scaling factors for each artifact
    """
    # Copy clean EEG
    contaminated_eeg = clean_eeg.copy()
    
    # If multiple artifacts, allocate equal SNR contribution for each
    individual_snr = target_snr + 10 * np.log10(len(artifact_types))
    
    alphas = {}
    total_artifact = np.zeros_like(clean_eeg)
    
    # Add each artifact
    for artifact_type in artifact_types:
        # Randomly select an artifact sample
        artifact_data = artifacts[artifact_type]
        artifact_idx = random.randint(0, len(artifact_data) - 1)
        artifact = artifact_data[artifact_idx]
        
        # Adjust artifact to achieve target SNR
        scaled_artifact, alpha = adjust_artifact_for_snr(clean_eeg, artifact, individual_snr)
        
        # Record scaling factor
        alphas[artifact_type] = alpha
        
        # Accumulate artifact
        total_artifact += scaled_artifact
    
    # Add artifact to clean EEG
    contaminated_eeg = clean_eeg + total_artifact
    
    return contaminated_eeg, alphas

def create_dataset(eeg_data, artifacts, artifact_combinations, samples_per_set=100000, snr_range=(-7, 2)):
    """
    Create dataset
    
    Args:
        eeg_data: Clean EEG data
        artifacts: Dictionary of artifacts
        artifact_combinations: List of artifact combinations
        samples_per_set: Number of samples per dataset
        snr_range: SNR range (dB)
    """
    # Create output directory
    output_base_dir = 'contaminated_eeg_datasets'
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Create dataset for each artifact combination
    for artifact_types in artifact_combinations:
        # Create combination name
        combination_name = '+'.join(artifact_types)
        print(f"\nGenerating {combination_name} artifact dataset...")
        
        # Create output directory
        output_dir = os.path.join(output_base_dir, combination_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data containers
        X_clean = []
        X_contaminated = []
        snrs = []
        alphas_list = []
        
        # Generate samples
        total_samples = samples_per_set  # Generate 20% extra samples for splitting
        
        for i in tqdm(range(int(total_samples)), desc=f"Generating {combination_name} samples"):
            # Randomly select a clean EEG sample
            eeg_idx = random.randint(0, len(eeg_data) - 1)
            clean_eeg = eeg_data[eeg_idx]
            
            # Randomly select SNR
            snr = random.uniform(snr_range[0], snr_range[1])
            
            # Generate contaminated EEG
            contaminated_eeg, alphas = generate_contaminated_eeg(
                clean_eeg, artifacts, artifact_types, target_snr=snr
            )
            
            # Store samples
            X_clean.append(clean_eeg)
            X_contaminated.append(contaminated_eeg)
            snrs.append(snr)
            alphas_list.append(alphas)
        
        # Convert to numpy arrays
        X_clean = np.array(X_clean)
        X_contaminated = np.array(X_contaminated)
        snrs = np.array(snrs)
        
        # Split dataset: 80% training, 10% validation, 10% testing
        # First split out the test set
        X_clean_temp, X_clean_test, X_contaminated_temp, X_contaminated_test, snrs_temp, snrs_test = train_test_split(
            X_clean, X_contaminated, snrs, test_size=0.1, random_state=42
        )
        
        # Then split out the validation set from the remaining data
        X_clean_train, X_clean_val, X_contaminated_train, X_contaminated_val, snrs_train, snrs_val = train_test_split(
            X_clean_temp, X_contaminated_temp, snrs_temp, test_size=0.1/0.9, random_state=42
        )
        
        # Modify the total number of samples and split ratio
        samples_per_set = 100000  # Total number of samples
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1

        # Calculate the number of samples for each set
        train_samples = int(samples_per_set * train_ratio)
        val_samples = int(samples_per_set * val_ratio)
        test_samples = int(samples_per_set * test_ratio)

        # Ensure each set has exact sample count
        X_clean_train = X_clean_train[:train_samples]
        X_contaminated_train = X_contaminated_train[:train_samples]
        snrs_train = snrs_train[:train_samples]

        X_clean_val = X_clean_val[:val_samples]
        X_contaminated_val = X_contaminated_val[:val_samples]
        snrs_val = snrs_val[:val_samples]

        X_clean_test = X_clean_test[:test_samples]
        X_contaminated_test = X_contaminated_test[:test_samples]
        snrs_test = snrs_test[:test_samples]
        
        # Save dataset
        print(f"Saving {combination_name} dataset...")
        
        # Training set
        np.save(os.path.join(output_dir, 'X_clean_train.npy'), X_clean_train)
        np.save(os.path.join(output_dir, 'X_contaminated_train.npy'), X_contaminated_train)
        np.save(os.path.join(output_dir, 'snrs_train.npy'), snrs_train)
        
        # Validation set
        np.save(os.path.join(output_dir, 'X_clean_val.npy'), X_clean_val)
        np.save(os.path.join(output_dir, 'X_contaminated_val.npy'), X_contaminated_val)
        np.save(os.path.join(output_dir, 'snrs_val.npy'), snrs_val)
        
        # Test set
        np.save(os.path.join(output_dir, 'X_clean_test.npy'), X_clean_test)
        np.save(os.path.join(output_dir, 'X_contaminated_test.npy'), X_contaminated_test)
        np.save(os.path.join(output_dir, 'snrs_test.npy'), snrs_test)
        
        print(f"{combination_name} dataset generation completed:")
        print(f"- Training set: {X_clean_train.shape[0]} samples")
        print(f"- Validation set: {X_clean_val.shape[0]} samples")
        print(f"- Test set: {X_clean_test.shape[0]} samples")
        print(f"- SNR range: {snrs_train.min():.2f} dB to {snrs_train.max():.2f} dB")

def main():
    # Load data
    eeg_data, artifacts = load_data()
    
    # Define artifact combinations
    artifact_combinations = [
        # Single artifact
        ['EMG'],
        ['EOG'],
        ['ECG'],
        # Mixed artifacts
        ['EMG', 'EOG'],
        ['EMG', 'ECG'],
        ['EOG', 'ECG'],
        ['EMG', 'EOG', 'ECG']
    ]
    
    # Create dataset
    create_dataset(eeg_data, artifacts, artifact_combinations, samples_per_set=100000, snr_range=(-7, 2))
    
    print("\nAll datasets generated successfully!")

if __name__ == "__main__":
    main()