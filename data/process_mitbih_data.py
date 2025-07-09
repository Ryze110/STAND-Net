import os
import numpy as np
import wfdb
import scipy.signal as signal
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_mitbih_record(record_name, data_dir='mit-bih-arrhythmia-database-1.0.0'):
    """
    Load a record from the MIT-BIH Arrhythmia Database
    
    Args:
        record_name: Record name, e.g., '100', '101', etc.
        data_dir: Database directory
    
    Returns:
        signals: ECG signal data
        fs: Sampling frequency
    """
    try:
        # Load the record
        record = wfdb.rdrecord(os.path.join(data_dir, record_name))
        signals = record.p_signal
        fs = record.fs
        return signals, fs
    except Exception as e:
        print(f"Error loading record {record_name}: {e}")
        return None, None

def process_ecg_signal(signal_data, original_fs=360, target_fs=256, lowpass_freq=45):
    """
    Process ECG signal: low-pass filtering and downsampling
    
    Args:
        signal_data: ECG signal data
        original_fs: Original sampling frequency
        target_fs: Target sampling frequency
        lowpass_freq: Low-pass filter cutoff frequency
    
    Returns:
        processed_signal: Processed signal
    """
    # Design low-pass filter
    nyquist_freq = original_fs / 2
    normalized_cutoff = lowpass_freq / nyquist_freq
    b, a = signal.butter(4, normalized_cutoff, 'low')
    
    # Apply low-pass filter
    filtered_signal = signal.filtfilt(b, a, signal_data, axis=0)
    
    # Downsample
    if original_fs != target_fs:
        # Calculate downsampling factor
        resampled_signal = signal.resample(filtered_signal, 
                                          int(len(filtered_signal) * target_fs / original_fs), 
                                          axis=0)
    else:
        resampled_signal = filtered_signal
    
    return resampled_signal

def segment_signal(signal_data, fs=256, segment_duration=2):
    """
    Segment the signal into fixed-duration segments
    
    Args:
        signal_data: Signal data
        fs: Sampling frequency
        segment_duration: Segment duration (seconds)
    
    Returns:
        segments: List of segmented signal
    """
    # Calculate the number of samples per segment
    samples_per_segment = int(fs * segment_duration)
    
    # Calculate the number of complete segments
    num_segments = len(signal_data) // samples_per_segment
    
    # Segment the signal
    segments = []
    for i in range(num_segments):
        start_idx = i * samples_per_segment
        end_idx = start_idx + samples_per_segment
        segment = signal_data[start_idx:end_idx]
        segments.append(segment)
    
    return np.array(segments)

def process_mitbih_database(data_dir='mit-bih-arrhythmia-database-1.0.0', output_dir='processed_ecg_artifacts'):
    """
    Process the entire MIT-BIH database
    
    Args:
        data_dir: Database directory
        output_dir: Output directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all record files
    record_files = [f.split('.')[0] for f in os.listdir(data_dir) 
                   if f.endswith('.hea') and not f.startswith('x_')]
    record_files = list(set(record_files))  # Remove duplicates
    
    all_segments = []
    
    for record_name in tqdm(record_files, desc="Processing records"):
        # Load the record
        signals, fs = load_mitbih_record(record_name, data_dir)
        if signals is None:
            continue
        
        # Process each lead
        for channel in range(signals.shape[1]):
            # Extract single-lead signal
            channel_signal = signals[:, channel]
            
            # Process signal: low-pass filtering and downsampling
            processed_signal = process_ecg_signal(channel_signal, fs, 256, 45)
            
            # Segment the signal
            segments = segment_signal(processed_signal, 256, 2)
            
            # Add to the overall list
            all_segments.append(segments)
    
    # Combine all segments
    all_segments = np.vstack(all_segments)
    
    # Save the processed data
    np.save(os.path.join(output_dir, 'ecg_artifacts.npy'), all_segments)
    
    print(f"Processing completed. Generated {len(all_segments)} ECG artifact segments.")
    print(f"Data saved to {os.path.join(output_dir, 'ecg_artifacts.npy')}")
    
    return all_segments

def visualize_samples(segments, num_samples=5, save_path=None):
    """
    Visualize processed ECG segment samples
    
    Args:
        segments: ECG segments
        num_samples: Number of samples to visualize
        save_path: Save path, if None, display the image
    """
    plt.figure(figsize=(15, 10))
    
    for i in range(min(num_samples, len(segments))):
        plt.subplot(num_samples, 1, i+1)
        
        # If multi-channel, display the first channel only
        if len(segments[i].shape) > 1:
            plt.plot(segments[i][:, 0])
        else:
            plt.plot(segments[i])
            
        plt.title(f'ECG Segment Sample #{i+1}')
        plt.xlabel('Sample Points')
        plt.ylabel('Amplitude')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Sample images saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Process MIT-BIH database
    ecg_artifacts = process_mitbih_database()
    
    # Visualize some samples
    visualize_samples(ecg_artifacts[:5], save_path='ecg_artifact_samples.png')