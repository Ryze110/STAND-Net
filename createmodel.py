from model_structure.STC_AutoEncoder_model import SNNAutoencoder


# --------------------------------
# Main Function
# --------------------------------

def create_model(model_type, input_size):
    """
    Create STAANet model
    
    Args:
        model_type: str, fixed as 'staanet'
        input_size: int, input sequence length
        
    Returns:
        PyTorch model
    """
    if model_type == 'staanet':
        # SNN autoencoder with attention mechanism
        return SNNAutoencoder(
            input_size=1,
            hidden_channels=[64, 128, 256, 512, 1024, 512, 256, 128, 64],
            kernel_size=15,
            tau=2.0,
            v_threshold=1.0,
            detach_reset=True,
            T=8
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only 'staanet' is supported")
