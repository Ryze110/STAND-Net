# STAANet: Spiking Temporal Attention Autoencoder Network for EEG Signal High-Quality and Power-Efficient Denoising


## Project Overview

STAANet is a deep learning framework based on Spiking Neural Networks (SNN) and attention mechanisms for EEG signal denoising. This project is specifically designed to remove various physiological artifacts from EEG signals, including EMG (electromyography), EOG (electrooculography), and ECG (electrocardiography) interference.

## Key Features

- **Spiking Neural Network Architecture**: Bio-inspired SNN model with superior temporal modeling capabilities and energy efficiency
- **Attention Mechanism**: Integrated spike-based attention mechanism that adaptively focuses on important temporal features
- **Multi-Artifact Support**: Supports removal of EMG, EOG, ECG artifacts and their combinations
- **Autoencoder Design**: End-to-end signal reconstruction using encoder-decoder architecture
- **Configurable Time Steps**: Supports customizable SNN simulation time steps
- **Comprehensive Evaluation**: Provides multiple evaluation metrics including MSE, MAE, RRMSE, SNR, PSNR, and correlation coefficient

## Model Structure




## Requirements
- **Python**:3.8
- **Pytorch**：2.0
- **Cuda**：11.8
- **spikingjelly**:0.14
- NumPy, Matplotlib, tqdm, scikit-learn, and other common scientific computing libraries

## How to start

Download the EEGDenoise dataset and MIT-BIH Arrhythmia Database from https://github.com/ncclabsustech/EEGdenoiseNet and https://www.physionet.org/content/mitdb/1.0.0/, and use the code in the data folder to generate a semi-simulated dataset for model training and testing. Modify the parameter settings in main.py according to the corresponding configuration to directly train STAANet.

  
