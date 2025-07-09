import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from units import device


# --------------------------------
# Data Loading and Preprocessing
# --------------------------------

def load_dataset(data_dir, artifact_type=None):
    """
    Load EEG dataset with noisy and clean signals.
    
    Args:
        data_dir: Base directory of the dataset
        artifact_type: Artifact type, e.g., 'EMG', 'EOG', 'ECG', 'EMG+EOG', etc.
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, snrs_test
        X: Contaminated EEG signals
        y: Clean EEG signals
        snrs_test: SNR values for the test set
    """
    print("Loading dataset...")
    
    # Use subdirectory if artifact type is specified
    if artifact_type:
        full_data_dir = os.path.join(data_dir, artifact_type)
    else:
        full_data_dir = data_dir
    
    try:
        # Load training data
        X_contaminated_train = np.load(os.path.join(full_data_dir, 'X_contaminated_train.npy'))
        X_clean_train = np.load(os.path.join(full_data_dir, 'X_clean_train.npy'))
        
        # Load validation data
        X_contaminated_val = np.load(os.path.join(full_data_dir, 'X_contaminated_val.npy'))
        X_clean_val = np.load(os.path.join(full_data_dir, 'X_clean_val.npy'))
        
        # Load test data
        X_contaminated_test = np.load(os.path.join(full_data_dir, 'X_contaminated_test.npy'))
        X_clean_test = np.load(os.path.join(full_data_dir, 'X_clean_test.npy'))
        
        # Load SNR information (if available)
        snrs_test = None
        try:
            snrs_test = np.load(os.path.join(full_data_dir, 'snrs_test.npy'))
            print(f"SNR information for test set: {snrs_test.shape}")
        except FileNotFoundError:
            print("SNR information file for test set not found, using calculated SNR values")
        
        print(f"Training set: {X_contaminated_train.shape}, {X_clean_train.shape}")
        print(f"Validation set: {X_contaminated_val.shape}, {X_clean_val.shape}")
        print(f"Test set: {X_contaminated_test.shape}, {X_clean_test.shape}")
        
        return X_contaminated_train, X_clean_train, X_contaminated_val, X_clean_val, X_contaminated_test, X_clean_test, snrs_test
    
    except Exception as e:
        print(f"Error occurred while loading dataset: {e}")
        return None, None, None, None, None, None, None

def prepare_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, snrs_test=None, batch_size=512):
    """
    Create PyTorch DataLoaders for training, validation, and testing.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        snrs_test: SNR values for test data
        batch_size: Batch size
    """
    # Reshape for CNN if needed [batch_size, channels, sequence_length]
    if len(X_train.shape) == 2:
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        y_train = y_train.reshape(y_train.shape[0], 1, y_train.shape[1])
        X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        y_val = y_val.reshape(y_val.shape[0], 1, y_val.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        y_test = y_test.reshape(y_test.shape[0], 1, y_test.shape[1])
    
    # Calculate dataset size
    print(f"Dataset size: {X_train.nbytes / 1e9:.2f} GB")
    use_gpu_data = X_train.nbytes / 1e9 < 5.0  # Preload to GPU if size is less than 5GB
    
    if use_gpu_data:
        # Preload data to GPU, must use single-process DataLoader
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.FloatTensor(y_val).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.FloatTensor(y_test).to(device)
        print("Data preloaded to GPU. Using single-process data loading.")
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Include SNR information in test set if available
        if snrs_test is not None:
            snrs_test_tensor = torch.FloatTensor(snrs_test).to(device)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor, snrs_test_tensor)
        else:
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Use single-process DataLoader for GPU data
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        # Keep data on CPU, use multi-process DataLoader
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        print("Data kept on CPU. Using multi-process data loading.")
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Include SNR information in test set if available
        if snrs_test is not None:
            snrs_test_tensor = torch.FloatTensor(snrs_test)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor, snrs_test_tensor)
        else:
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Use multi-process DataLoader for CPU data
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)
    
    return train_loader, val_loader, test_loader

# Add normalization function
def data_normalize(clean_signal, noisy_signal, mean_norm=False):
    """
    Apply normalization method similar to original EEGDnet
    
    Args:
        clean_signal: Clean EEG signals [samples, (channels), length]
        noisy_signal: Noisy EEG signals [samples, (channels), length]
        mean_norm: Whether to apply mean normalization, default is False
        
    Returns:
        clean_norm: Normalized clean signals
        noisy_norm: Normalized noisy signals
        std_values: Standard deviation values used for normalization (can be used for denormalization later)
    """
    # Ensure inputs are numpy arrays
    if isinstance(clean_signal, torch.Tensor):
        clean_signal = clean_signal.cpu().numpy()
    if isinstance(noisy_signal, torch.Tensor):
        noisy_signal = noisy_signal.cpu().numpy()
    
    # Save original shapes
    clean_shape = clean_signal.shape
    noisy_shape = noisy_signal.shape
    
    # Handle 3D data [batch, channels, length]
    if len(clean_shape) == 3:
        # Reshape to 2D [batch, length], remove channel dimension
        clean_signal = clean_signal.reshape(clean_shape[0], clean_shape[2])
        noisy_signal = noisy_signal.reshape(noisy_shape[0], noisy_shape[2])
    
    # Calculate mean (if needed)
    if mean_norm:
        mean_values = np.mean(noisy_signal, axis=1, keepdims=True)
    else:
        mean_values = np.zeros((noisy_signal.shape[0], 1))
    
    # Calculate standard deviation (along signal length)
    std_values = np.std(noisy_signal, axis=1, keepdims=True)
    
    # Normalize signals
    clean_norm = (clean_signal - mean_values) / std_values
    noisy_norm = (noisy_signal - mean_values) / std_values
    
    # Restore shape if original data was 3D
    if len(clean_shape) == 3:
        clean_norm = clean_norm.reshape(clean_shape)
        noisy_norm = noisy_norm.reshape(noisy_shape)
    
    print(f"Normalization applied: Original shape={clean_shape}, Std range={np.min(std_values):.4f} to {np.max(std_values):.4f}")
    
    return clean_norm, noisy_norm, std_values

def load_and_normalize_dataset(data_dir, artifact_type=None, use_data_norm=True):
    """
    Load dataset and apply normalization
    
    Args:
        data_dir: Base directory of the dataset
        artifact_type: Artifact type, e.g., 'EMG', 'EOG', 'ECG', etc.
        
    Returns:
        Normalized training, validation, and test datasets
    """
    # First, load the raw data
    X_train, y_train, X_val, y_val, X_test, y_test, snrs_test = load_dataset(data_dir, artifact_type)
    
    if X_train is None:
        return None, None, None, None, None, None, None
    
    # Check if EEGDnet normalization needs to be applied
    if use_data_norm:
        # Normalize training set
        y_train_norm, X_train_norm, train_std = data_normalize(y_train, X_train)
        
        # Normalize validation set
        y_val_norm, X_val_norm, val_std = data_normalize(y_val, X_val)
        
        # Normalize test set
        y_test_norm, X_test_norm, test_std = data_normalize(y_test, X_test)
        
        return X_train_norm, y_train_norm, X_val_norm, y_val_norm, X_test_norm, y_test_norm, snrs_test
    else:
        # Return raw data without normalization
        return X_train, y_train, X_val, y_val, X_test, y_test, snrs_test