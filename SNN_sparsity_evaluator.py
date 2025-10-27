# Neuron Sparsity Evaluation for SNN Networks

import torch
import numpy as np
from tqdm import tqdm
import spikingjelly.clock_driven.functional as functional

class SNNSparsityEvaluator:
    """Used to calculate the activation sparsity of SNN models"""
    
    def __init__(self, snn_model, test_loader, device, T=10):
        """
        Initialize the evaluator
        
        Args:
            snn_model: Trained SNN model
            test_loader: Test data loader
            device: Computing device
            T: Simulation time steps for SNN
        """
        self.snn_model = snn_model
        self.test_loader = test_loader
        self.device = device
        self.T = T
        
        # Used to store statistical results
        self.statistics = {
            'spike_rates': [],
            'layer_spike_rates': {}
        }
    
    def evaluate_sparsity(self, num_batches=None):
        """
        Evaluate the spike sparsity of SNN model
        
        Args:
            num_batches: Number of batches to evaluate, None means all
            
        Returns:
            Dictionary containing overall and per-layer sparsity
        """
        print("\n===== Starting SNN Neuron Activation Sparsity Evaluation =====")
        
        # Evaluate SNN model
        self.snn_model.eval()
        
        # Collect all LIF neuron layers
        lif_layers = {}
        
        def find_lif_neurons(module, prefix=''):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                if hasattr(child, 'lif'):
                    lif_layers[full_name] = child.lif
                    self.statistics['layer_spike_rates'][full_name] = []
                else:
                    find_lif_neurons(child, full_name)
        
        find_lif_neurons(self.snn_model)
        print(f"Found {len(lif_layers)} LIF neuron layers")
        
        # Use hooks to directly collect spike data
        layer_spike_counts = {}
        layer_neuron_counts = {}
        
        # Define hook function to capture spikes
        def spike_hook(name):
            def hook(module, input, output):
                # For SNN, output is usually spikes (0 or 1)
                # If not boolean, convert to spikes through thresholding
                if output.dtype != torch.bool and output.dtype != torch.uint8:
                    spikes = (output > 0).float()
                else:
                    spikes = output.float()
                
                # Calculate total number of spikes in this time step
                spike_count = spikes.sum().item()
                # Calculate total number of neurons
                neuron_count = np.prod(spikes.shape)
                
                # Accumulate to layer statistics
                if name not in layer_spike_counts:
                    layer_spike_counts[name] = 0
                    layer_neuron_counts[name] = 0
                    
                layer_spike_counts[name] += spike_count
                layer_neuron_counts[name] += neuron_count
            
            return hook
        
        # Register hooks
        hooks = []
        for name, lif_layer in lif_layers.items():
            hook = lif_layer.register_forward_hook(spike_hook(name))
            hooks.append(hook)
        
        # Process data batches
        batch_count = 0
        progress = tqdm(self.test_loader, desc="Calculating SNN Sparsity")
        
        for inputs, _ in progress:
            inputs = inputs.to(self.device)
            
            # Clear statistics for each batch
            layer_spike_counts.clear()
            layer_neuron_counts.clear()
            
            # Reset SNN state
            functional.reset_net(self.snn_model)
            
            # Run one time series
            for t in range(self.T):
                with torch.no_grad():
                    _ = self.snn_model(inputs)
            
            # Calculate spike rate for each layer
            batch_total_spikes = 0
            batch_total_neurons = 0
            
            for name in lif_layers.keys():
                if name in layer_spike_counts and layer_neuron_counts[name] > 0:
                    # Calculate layer spike rate (average spikes per neuron per time step)
                    layer_spike_rate = layer_spike_counts[name] / layer_neuron_counts[name]
                    self.statistics['layer_spike_rates'][name].append(layer_spike_rate)
                    
                    batch_total_spikes += layer_spike_counts[name]
                    batch_total_neurons += layer_neuron_counts[name]
            
            # Calculate overall batch spike rate
            if batch_total_neurons > 0:
                batch_spike_rate = batch_total_spikes / batch_total_neurons
                self.statistics['spike_rates'].append(batch_spike_rate)
                progress.set_postfix(spike_rate=f"{batch_spike_rate:.6f}")
            
            batch_count += 1
            if num_batches is not None and batch_count >= num_batches:
                break
        
        # Remove all hooks
        for hook in hooks:
            hook.remove()
        
        # Check if any data was collected
        if not self.statistics['spike_rates']:
            print("Warning: No spike rate data collected. Trying alternative method...")
            
            # Directly check spikes of each LIF neuron
            class SpikesCollector:
                def __init__(self):
                    self.counts = {}
                    self.total_spikes = 0
                    self.total_neurons = 0
                
                def collect(self, name, spikes):
                    spike_count = spikes.sum().item()
                    neuron_count = np.prod(spikes.shape)
                    
                    if name not in self.counts:
                        self.counts[name] = {
                            'spikes': 0,
                            'neurons': 0
                        }
                    
                    self.counts[name]['spikes'] += spike_count
                    self.counts[name]['neurons'] += neuron_count
                    
                    self.total_spikes += spike_count
                    self.total_neurons += neuron_count
            
            collector = SpikesCollector()
            
            # Re-register hooks
            hooks = []
            for name, lif_layer in lif_layers.items():
                def make_hook(layer_name):
                    def hook(module, input, output):
                        if output.dtype != torch.bool and output.dtype != torch.uint8:
                            spikes = (output > 0).float()
                        else:
                            spikes = output.float()
                        collector.collect(layer_name, spikes)
                    return hook
                
                hooks.append(lif_layer.register_forward_hook(make_hook(name)))
            
            # Process one batch of data
            print("Running alternative method to collect data...")
            functional.reset_net(self.snn_model)
            
            test_batch = next(iter(self.test_loader))
            inputs = test_batch[0].to(self.device)
            
            # Run one time series
            for t in range(self.T):
                with torch.no_grad():
                    _ = self.snn_model(inputs)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Calculate spike rate
            if collector.total_neurons > 0:
                avg_spike_rate = collector.total_spikes / collector.total_neurons
                self.statistics['spike_rates'] = [avg_spike_rate]
                
                # Calculate per-layer spike rate
                for name, data in collector.counts.items():
                    if data['neurons'] > 0:
                        layer_rate = data['spikes'] / data['neurons']
                        self.statistics['layer_spike_rates'][name] = [layer_rate]
            else:
                print("Warning: Still unable to collect spike data. May need to check model structure.")
                return {
                    'overall_spike_rate': 0,
                    'overall_sparsity': 1,
                    'layer_spike_rates': {},
                    'layer_sparsity': {}
                }
        
        # Calculate average spike rate
        avg_spike_rate = np.mean(self.statistics['spike_rates'])
        print(f"SNN Overall Average Spike Rate: {avg_spike_rate:.6f} (average spikes per neuron per time step)")
        print(f"SNN Overall Sparsity: {1 - avg_spike_rate:.6f}")
        
        # Calculate layer-level average spike rate
        layer_sparsity = {}
        for name, rates in self.statistics['layer_spike_rates'].items():
            if rates:  # Ensure there is data
                layer_avg_rate = np.mean(rates)
                layer_sparsity[name] = 1 - layer_avg_rate
                print(f"Layer {name} Average Spike Rate: {layer_avg_rate:.6f}, Sparsity: {layer_sparsity[name]:.6f}")
        
        return {
            'overall_spike_rate': avg_spike_rate,
            'overall_sparsity': 1 - avg_spike_rate,
            'layer_spike_rates': {name: np.mean(rates) if rates else 0 for name, rates in self.statistics['layer_spike_rates'].items()},
            'layer_sparsity': layer_sparsity
        }


def evaluate_snn_sparsity(snn_model, test_loader, device, T=10, num_batches=10):
    """
    Evaluate the activation sparsity of SNN model
    
    Args:
        snn_model: Trained SNN model
        test_loader: Test data loader
        device: Computation device
        T: SNN timesteps
        num_batches: Number of batches to evaluate
    
    Returns:
        statistics: Dictionary containing sparsity evaluation results
    """
    # Create evaluator
    evaluator = SNNSparsityEvaluator(snn_model, test_loader, device, T)
    
    # Evaluate sparsity
    sparsity_stats = evaluator.evaluate_sparsity(num_batches)
    
    return sparsity_stats


# Example usage
if __name__ == "__main__":
    import sys
    import os
    
    # Add project root directory to Python path to ensure model_train module can be imported
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Import your SNN model and data loading code here
    try:
        from loaddata import load_dataset, prepare_data_loaders
        from model_structure.STC_AutoEncoder_model import SNNAutoencoder
    except ImportError:
        print("Unable to import necessary modules. Please ensure your SNN model and data loading functions are available.")
        sys.exit(1)

    # Check if CUDA is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define parameters (can also use argparse to parse command line arguments)
    model_path = 'snn_model_results_0320/snn_autoencoder.pth'
    data_dir = 'eeg_denoise_dataset_0319'
    batch_size = 64
    time_steps = 10
    num_batches = 10

    # Load dataset
    print("Loading dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test, _ = load_dataset(data_dir)
    _, _, test_loader = prepare_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch_size
    )

    # Create and load model
    print(f"Loading SNN model weights: {model_path}")
    model = SNNAutoencoder(
        input_size=1,
        hidden_channels=[64, 128, 256, 512, 1024, 512, 256, 128, 64],
        kernel_size=15,
        tau=2.0,
        v_threshold=1.0,
        detach_reset=True
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Evaluate model sparsity
    print(f"Starting SNN model sparsity evaluation...")
    sparsity_stats = evaluate_snn_sparsity(
        model, test_loader, device, time_steps, num_batches
    )
    
    print("\nEvaluation completed!")
    print(f"Overall sparsity: {sparsity_stats['overall_sparsity']:.6f}")