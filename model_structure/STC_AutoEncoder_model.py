import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from spikingjelly.clock_driven import neuron, layer, surrogate, functional
from model_structure.SNN_Attention import SNNSpikeAttention

class SNNResidualBlock(nn.Module):
    """Residual block for the decoder without attention mechanism"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, dropout=0.2,
                 tau=2.0, v_threshold=1.0, detach_reset=True):
        super(SNNResidualBlock, self).__init__()
        
        # First SNN convolution block
        self.snn_conv1 = SNNConvBlock(
            in_channels, out_channels, kernel_size, dilation, dropout,
            tau, v_threshold, detach_reset
        )
        
        # Second SNN convolution block
        self.snn_conv2 = SNNConvBlock(
            out_channels, out_channels, kernel_size, dilation, dropout,
            tau, v_threshold, detach_reset
        )
        
        # Add 1x1 convolution for channel conversion if input and output channels differ
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x, T=8):
        residual = x
        
        # First SNN convolution
        out = self.snn_conv1(x, T)
        
        # Second SNN convolution
        out = self.snn_conv2(out, T)
        
        # Residual connection
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        # Add residual
        return out + residual
    
    def reset(self):
        """Reset all neuron states"""
        self.snn_conv1.lif.reset()
        self.snn_conv2.lif.reset()

class SNNResidualBlockWithSpikeAttention(nn.Module):
    """Residual block for the encoder with spike attention mechanism"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, dropout=0.2,
                 tau=2.0, v_threshold=1.0, detach_reset=True):
        super(SNNResidualBlockWithSpikeAttention, self).__init__()
        
        # First SNN convolution block
        self.snn_conv1 = SNNConvBlock(
            in_channels, out_channels, kernel_size, dilation, dropout,
            tau, v_threshold, detach_reset
        )
        
        # Second SNN convolution block
        self.snn_conv2 = SNNConvBlock(
            out_channels, out_channels, kernel_size, dilation, dropout,
            tau, v_threshold, detach_reset
        )
        
        # Spike attention mechanism
        self.attention = SNNSpikeAttention(out_channels)
        
        # Add 1x1 convolution for channel conversion if input and output channels differ
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x, T=8):
        residual = x
        
        # First SNN convolution
        out = self.snn_conv1(x, T)
        
        # Second SNN convolution, return spike rates
        out, spike_rates = self.snn_conv2(out, T, return_spike_rates=True)
        
        # Apply spike attention mechanism
        out = self.attention(out, spike_rates)
        
        # Residual connection
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        # Add residual
        return out + residual
    
    def reset(self):
        """Reset all neuron states"""
        self.snn_conv1.lif.reset()
        self.snn_conv2.lif.reset()

class SNNConvBlock(nn.Module):
    """SNN convolution block with convolution, batch normalization, and LIF neurons"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, dropout=0.2, 
                 tau=2.0, v_threshold=1.0, detach_reset=True):
        super(SNNConvBlock, self).__init__()
        
        self.padding = (kernel_size-1) * dilation // 2
        
        # Convolution layer
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        
        # LIF neurons with surrogate gradient for backpropagation
        self.lif = neuron.LIFNode(
            tau=tau,                        # Membrane potential time constant
            v_threshold=v_threshold,        # Firing threshold
            surrogate_function=surrogate.ATan(),  # Surrogate gradient function
            detach_reset=detach_reset       # Whether to detach reset gradient
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, T=8, return_spike_rates=False):
        """
        Forward propagation, convert simulated input to spike sequences and process with convolution
        x: Input features [batch, channels, length]
        T: Simulation time steps
        return_spike_rates: Whether to return spike rates
        """
        # Convolution processing
        conv_out = self.conv(x)
        conv_out = self.bn(conv_out)
        
        # Simulate over T time steps
        spk_out = 0.
        spike_rates = 0.
        self.lif.reset()  # Reset neuron states
        
        for t in range(T):
            # Simulate neurons for each time step
            spk = self.lif(conv_out)
            spk_out = spk_out + spk  # Accumulate spike output
            spike_rates = spike_rates + spk  # Accumulate spike rates
        
        # Average spike output and apply dropout
        out = spk_out / T
        out = self.dropout(out)
        
        if return_spike_rates:
            return out, spike_rates / T
        return out

class SNNAutoencoder(nn.Module):
    """SNN autoencoder with spike attention in the encoder and regular residual blocks in the decoder"""
    def __init__(self, input_size=1, hidden_channels=[32, 64, 128, 64, 32], kernel_size=15,
                 tau=2.0, v_threshold=1.0, detach_reset=True, T=8):
        super(SNNAutoencoder, self).__init__()
        
        self.T = T  # Save time steps as a class attribute
        
        # Create encoder and decoder layers
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        num_levels = len(hidden_channels)
        
        # Encoder part - use residual blocks with spike attention
        in_channels = input_size
        for i in range(num_levels // 2 + 1):
            out_channels = hidden_channels[i]
            dilation = 2 ** i  # Exponentially increasing dilation rate
            self.encoder_layers.append(SNNResidualBlockWithSpikeAttention(
                in_channels, out_channels, kernel_size, dilation,
                tau=tau, v_threshold=v_threshold, detach_reset=detach_reset
            ))
            in_channels = out_channels
        
        # Decoder part - use regular residual blocks
        for i in range(num_levels // 2 + 1, num_levels):
            out_channels = hidden_channels[i]
            dilation = 2 ** (num_levels - i - 1)  # Decreasing dilation rate
            self.decoder_layers.append(SNNResidualBlock(
                in_channels, out_channels, kernel_size, dilation,
                tau=tau, v_threshold=v_threshold, detach_reset=detach_reset
            ))
            in_channels = out_channels
        
        # Final output layer to restore dimensions (no spike neurons)
        self.output_layer = nn.Conv1d(hidden_channels[-1], input_size, kernel_size=1)
        
    def forward(self, x, T=None):
        """
        Forward propagation
        x: Input features [batch, channels, length]
        T: Simulation time steps, use initialized value if None
        """
        if T is None:
            T = self.T
            
        # Process input shape
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add channel dimension
            
        # Encoding phase
        for layer in self.encoder_layers:
            x = layer(x, T)
            
        # Decoding phase
        for layer in self.decoder_layers:
            x = layer(x, T)
        
        # Output layer (no spike neurons)
        output = self.output_layer(x)
        
        return output
    
    def reset(self):
        """Reset all neuron states"""
        for layer in self.encoder_layers:
            layer.reset()
        for layer in self.decoder_layers:
            layer.reset()