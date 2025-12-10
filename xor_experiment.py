"""
XOR Learning Experiment: Comparing Muon vs AdamW Optimizers

This script trains a simple feedforward neural network to learn the XOR function,
comparing the performance of Muon and AdamW optimizers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from muon import SingleDeviceMuonWithAuxAdam


# XOR Dataset
XOR_INPUTS = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=torch.float32)
XOR_TARGETS = torch.tensor([[0.], [1.], [1.], [0.]], dtype=torch.float32)


class SimpleNet(nn.Module):
    """Simple feedforward network for XOR problem"""
    def __init__(self, hidden_size=2):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


def train_model(optimizer_name, lr, num_epochs=1000, hidden_size=2, seed=42):
    """
    Train a model with specified optimizer
    
    Args:
        optimizer_name: 'muon' or 'adamw'
        lr: learning rate
        num_epochs: number of training epochs
        hidden_size: number of hidden units
        seed: random seed for reproducibility
        
    Returns:
        Dictionary containing training history and final model
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create model
    model = SimpleNet(hidden_size=hidden_size)
    
    # Create optimizer
    if optimizer_name.lower() == 'muon':
        # Muon works best with 2D parameters (weight matrices)
        # For 1D parameters (biases), we use auxiliary AdamW
        # Separate parameters into 2D (weights) and 1D (biases)
        muon_params = [p for p in model.parameters() if p.ndim >= 2]
        adam_params = [p for p in model.parameters() if p.ndim < 2]
        
        param_groups = []
        if muon_params:
            param_groups.append({
                'params': muon_params,
                'lr': lr,
                'use_muon': True
            })
        if adam_params:
            param_groups.append({
                'params': adam_params,
                'lr': lr * 15,  # AdamW typically needs higher LR with Muon's default
                'use_muon': False
            })
        
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    elif optimizer_name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Training history
    history = {
        'loss': [],
        'accuracy': [],
        'epoch': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(XOR_INPUTS)
        loss = F.binary_cross_entropy(outputs, XOR_TARGETS)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        predictions = (outputs > 0.5).float()
        accuracy = (predictions == XOR_TARGETS).float().mean().item()
        
        # Record history
        if epoch % 50 == 0 or epoch == num_epochs - 1:
            history['loss'].append(loss.item())
            history['accuracy'].append(accuracy)
            history['epoch'].append(epoch)
            
            if epoch % 500 == 0:
                print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f} | Accuracy: {accuracy:.4f}")
    
    return {
        'model': model,
        'history': history,
        'optimizer_name': optimizer_name,
        'final_loss': history['loss'][-1],
        'final_accuracy': history['accuracy'][-1]
    }


def run_experiment(num_trials=5, num_epochs=5000):
    """
    Run comparison experiment between Muon and AdamW
    
    Args:
        num_trials: number of trials to run for each optimizer
        num_epochs: number of training epochs per trial
    """
    print("="*70)
    print("XOR Learning Experiment: Muon vs AdamW")
    print("="*70)
    print()
    
    # Different learning rates to try
    muon_lr = 0.05  # Default Muon learning rate
    adamw_lr = 0.01  # Typical AdamW learning rate
    
    results = {
        'muon': [],
        'adamw': []
    }
    
    # Run trials
    for trial in range(num_trials):
        print(f"\n{'='*70}")
        print(f"Trial {trial + 1}/{num_trials}")
        print(f"{'='*70}")
        
        # Train with Muon
        print(f"\n--- Training with Muon (lr={muon_lr}) ---")
        muon_result = train_model('muon', lr=muon_lr, num_epochs=num_epochs, seed=trial)
        results['muon'].append(muon_result)
        
        # Train with AdamW
        print(f"\n--- Training with AdamW (lr={adamw_lr}) ---")
        adamw_result = train_model('adamw', lr=adamw_lr, num_epochs=num_epochs, seed=trial)
        results['adamw'].append(adamw_result)
    
    return results


def plot_results(results, save_path='xor_comparison.png'):
    """
    Plot comparison of Muon vs AdamW training dynamics
    
    Args:
        results: Dictionary containing results for both optimizers
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss curves
    ax = axes[0]
    for i, result in enumerate(results['muon']):
        epochs = result['history']['epoch']
        loss = result['history']['loss']
        ax.plot(epochs, loss, color='#FF6B6B', alpha=0.3, linewidth=1)
    
    for i, result in enumerate(results['adamw']):
        epochs = result['history']['epoch']
        loss = result['history']['loss']
        ax.plot(epochs, loss, color='#4ECDC4', alpha=0.3, linewidth=1)
    
    # Plot average curves
    muon_avg_loss = np.mean([r['history']['loss'] for r in results['muon']], axis=0)
    adamw_avg_loss = np.mean([r['history']['loss'] for r in results['adamw']], axis=0)
    epochs = results['muon'][0]['history']['epoch']
    
    ax.plot(epochs, muon_avg_loss, color='#FF6B6B', linewidth=2.5, label='Muon (avg)', marker='o', markersize=3)
    ax.plot(epochs, adamw_avg_loss, color='#4ECDC4', linewidth=2.5, label='AdamW (avg)', marker='s', markersize=3)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (BCE)', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot accuracy curves
    ax = axes[1]
    for i, result in enumerate(results['muon']):
        epochs = result['history']['epoch']
        accuracy = result['history']['accuracy']
        ax.plot(epochs, accuracy, color='#FF6B6B', alpha=0.3, linewidth=1)
    
    for i, result in enumerate(results['adamw']):
        epochs = result['history']['epoch']
        accuracy = result['history']['accuracy']
        ax.plot(epochs, accuracy, color='#4ECDC4', alpha=0.3, linewidth=1)
    
    # Plot average curves
    muon_avg_acc = np.mean([r['history']['accuracy'] for r in results['muon']], axis=0)
    adamw_avg_acc = np.mean([r['history']['accuracy'] for r in results['adamw']], axis=0)
    
    ax.plot(epochs, muon_avg_acc, color='#FF6B6B', linewidth=2.5, label='Muon (avg)', marker='o', markersize=3)
    ax.plot(epochs, adamw_avg_acc, color='#4ECDC4', linewidth=2.5, label='AdamW (avg)', marker='s', markersize=3)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.show()


def print_summary(results):
    """Print summary statistics of the experiment"""
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    for opt_name in ['muon', 'adamw']:
        opt_results = results[opt_name]
        final_losses = [r['final_loss'] for r in opt_results]
        final_accs = [r['final_accuracy'] for r in opt_results]
        
        print(f"\n{opt_name.upper()}:")
        print(f"  Final Loss:     {np.mean(final_losses):.6f} ± {np.std(final_losses):.6f}")
        print(f"  Final Accuracy: {np.mean(final_accs):.4f} ± {np.std(final_accs):.4f}")
        print(f"  Success Rate:   {sum(acc == 1.0 for acc in final_accs)}/{len(final_accs)} trials")
    
    # Test final predictions
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS (from first trial)")
    print("="*70)
    
    for opt_name in ['muon', 'adamw']:
        model = results[opt_name][0]['model']
        model.eval()
        with torch.no_grad():
            outputs = model(XOR_INPUTS)
        
        print(f"\n{opt_name.upper()} predictions:")
        print("  Input  | Target | Prediction | Output")
        print("  " + "-"*42)
        for i in range(len(XOR_INPUTS)):
            inp = XOR_INPUTS[i].numpy()
            target = XOR_TARGETS[i].item()
            output = outputs[i].item()
            pred = int(output > 0.5)
            print(f"  {inp}  |   {target:.0f}    |     {pred}      | {output:.4f}")




if __name__ == "__main__":
    # Run the experiment
    results = run_experiment(num_trials=5, num_epochs=800)
    
    # Plot results
    plot_results(results)
    
    # Print summary
    print_summary(results)

