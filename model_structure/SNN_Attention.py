import torch
import torch.nn as nn
import torch.nn.functional as F


class SNNSpikeAttention(nn.Module):
    """Attention mechanism based on spike activity"""
    def __init__(self, channels, kernel_size=7):
        super(SNNSpikeAttention, self).__init__()
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, 
            padding=(kernel_size-1)//2, bias=False
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, spike_rates):
        """
        Forward propagation for spike-based attention
        
        Args:
            x: Features [batch, channels, length]
            spike_rates: Spike rates [batch, channels, length]
        
        Returns:
            Attention-weighted features
        """
        # Compute average spike rate for each channel
        avg_spike = torch.mean(spike_rates, dim=1, keepdim=True)
        
        # Learn local dependencies through convolution
        attn = self.conv(avg_spike)
        
        # Apply sigmoid to obtain attention weights
        attn = self.sigmoid(attn)
        
        # Apply attention weights
        return x * attn.expand_as(x)
