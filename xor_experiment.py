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


def plot_loss_surface(model=None, grid_size=50, extent=1.5, save_path='loss_surface.png'):
    """
    Plot the loss surface around a model's parameters
    
    Creates a 2D slice through parameter space using two random directions.
    Based on the approach from "Visualizing the Loss Landscape of Neural Nets" (Li et al.)
    
    Args:
        model: Trained model to center the visualization on (if None, uses random init)
        grid_size: Number of points in each direction
        extent: How far to explore in each direction (in parameter space)
        save_path: Path to save the plot
    """
    print("\n" + "="*70)
    print("PLOTTING LOSS SURFACE")
    print("="*70)
    
    # Use provided model or create a new one
    if model is None:
        model = SimpleNet(hidden_size=2)
    
    # Get all parameters as a flat vector (center point)
    center_params = []
    for p in model.parameters():
        center_params.append(p.data.clone().flatten())
    center_params = torch.cat(center_params)
    
    # Create two random direction vectors (normalized)
    torch.manual_seed(42)
    direction1 = torch.randn_like(center_params)
    direction1 = direction1 / torch.norm(direction1)
    
    direction2 = torch.randn_like(center_params)
    # Make direction2 orthogonal to direction1
    direction2 = direction2 - (direction2 @ direction1) * direction1
    direction2 = direction2 / torch.norm(direction2)
    
    # Create grid
    alphas = np.linspace(-extent, extent, grid_size)
    betas = np.linspace(-extent, extent, grid_size)
    
    # Storage for losses
    losses = np.zeros((grid_size, grid_size))
    
    print(f"Computing loss surface ({grid_size}x{grid_size} = {grid_size**2} points)...")
    
    # Compute loss at each grid point
    def set_params(alpha, beta):
        """Set model parameters to center + alpha*dir1 + beta*dir2"""
        new_params = center_params + alpha * direction1 + beta * direction2
        
        # Unflatten and assign back to model
        offset = 0
        for p in model.parameters():
            numel = p.numel()
            p.data.copy_(new_params[offset:offset+numel].reshape(p.shape))
            offset += numel
    
    # Compute losses
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            set_params(alpha, beta)
            model.eval()
            with torch.no_grad():
                outputs = model(XOR_INPUTS)
                loss = F.binary_cross_entropy(outputs, XOR_TARGETS)
                losses[i, j] = loss.item()
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{grid_size} rows completed")
    
    # Restore original parameters
    set_params(0, 0)
    
    print("Creating visualization...")
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Contour plot
    X, Y = np.meshgrid(alphas, betas)
    levels = np.logspace(np.log10(losses.min() + 1e-8), np.log10(losses.max()), 20)
    contour = ax1.contourf(X, Y, losses.T, levels=levels, cmap='viridis', norm='log')
    ax1.contour(X, Y, losses.T, levels=levels, colors='white', alpha=0.3, linewidths=0.5)
    ax1.plot(0, 0, 'r*', markersize=15, label='Model parameters')
    ax1.set_xlabel('Direction 1 (α)', fontsize=12)
    ax1.set_ylabel('Direction 2 (β)', fontsize=12)
    ax1.set_title('Loss Surface (Contour)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    cbar1 = plt.colorbar(contour, ax=ax1)
    cbar1.set_label('Loss (BCE)', fontsize=11)
    
    # 3D surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X, Y, losses.T, cmap='viridis', 
                            norm='log', alpha=0.9, antialiased=True)
    ax2.plot([0], [0], [losses[grid_size//2, grid_size//2]], 'r*', 
             markersize=15, label='Model parameters')
    ax2.set_xlabel('Direction 1 (α)', fontsize=11)
    ax2.set_ylabel('Direction 2 (β)', fontsize=11)
    ax2.set_zlabel('Loss (BCE)', fontsize=11)
    ax2.set_title('Loss Surface (3D)', fontsize=14, fontweight='bold')
    ax2.view_init(elev=25, azim=45)
    cbar2 = plt.colorbar(surf, ax=ax2, shrink=0.6)
    cbar2.set_label('Loss (BCE)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nLoss surface plot saved to: {save_path}")
    plt.show()
    
    # Print statistics
    print(f"\nLoss Surface Statistics:")
    print(f"  Minimum loss: {losses.min():.6f}")
    print(f"  Maximum loss: {losses.max():.6f}")
    print(f"  Loss at center: {losses[grid_size//2, grid_size//2]:.6f}")
    print(f"  Loss range: {losses.max() - losses.min():.6f}")


def plot_complete_loss_surface(model=None, grid_size=100, extent=3.0, save_path='complete_loss_surface.png'):
    """
    Plot the COMPLETE loss surface by varying two specific weights
    
    Since we only have 2 hidden units, we can plot the actual loss surface
    by varying two specific weights (the first two weights of fc1) while
    keeping all other parameters fixed. This gives us a true loss surface
    rather than a random 2D projection.
    
    Args:
        model: Trained model to center the visualization on (if None, uses random init)
        grid_size: Number of points in each direction
        extent: How far to explore in each direction (absolute weight values)
        save_path: Path to save the plot
    """
    print("\n" + "="*70)
    print("PLOTTING COMPLETE LOSS SURFACE (2 specific weights)")
    print("="*70)
    
    # Use provided model or create a new one
    if model is None:
        model = SimpleNet(hidden_size=2)
    
    # Store original parameters
    original_params = {}
    for name, param in model.named_parameters():
        original_params[name] = param.data.clone()
    
    # We'll vary fc1.weight[0, 0] and fc1.weight[0, 1]
    # (the two weights connecting to the first hidden unit)
    center_w00 = original_params['fc1.weight'][0, 0].item()
    center_w01 = original_params['fc1.weight'][0, 1].item()
    
    print(f"Center point: fc1.weight[0,0]={center_w00:.4f}, fc1.weight[0,1]={center_w01:.4f}")
    print(f"Exploring range: [{center_w00-extent:.2f}, {center_w00+extent:.2f}] x [{center_w01-extent:.2f}, {center_w01+extent:.2f}]")
    
    # Create grid
    w00_values = np.linspace(center_w00 - extent, center_w00 + extent, grid_size)
    w01_values = np.linspace(center_w01 - extent, center_w01 + extent, grid_size)
    
    # Storage for losses
    losses = np.zeros((grid_size, grid_size))
    
    print(f"Computing loss surface ({grid_size}x{grid_size} = {grid_size**2} points)...")
    
    # Compute loss at each grid point
    for i, w00 in enumerate(w00_values):
        for j, w01 in enumerate(w01_values):
            # Set the two weights we're varying
            model.fc1.weight.data[0, 0] = w00
            model.fc1.weight.data[0, 1] = w01
            
            # Compute loss
            model.eval()
            with torch.no_grad():
                outputs = model(XOR_INPUTS)
                loss = F.binary_cross_entropy(outputs, XOR_TARGETS)
                losses[i, j] = loss.item()
        
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{grid_size} rows completed")
    
    # Restore original parameters
    for name, param in model.named_parameters():
        param.data.copy_(original_params[name])
    
    print("Creating visualization...")
    
    # Create the plot
    fig = plt.figure(figsize=(18, 6))
    
    # Contour plot
    ax1 = plt.subplot(131)
    X, Y = np.meshgrid(w00_values, w01_values)
    
    # Use logarithmic levels if loss range is large
    loss_min, loss_max = losses.min(), losses.max()
    if loss_max / loss_min > 100:
        levels = np.logspace(np.log10(loss_min + 1e-8), np.log10(loss_max), 30)
        contour = ax1.contourf(X, Y, losses.T, levels=levels, cmap='viridis', norm='log')
    else:
        levels = 30
        contour = ax1.contourf(X, Y, losses.T, levels=levels, cmap='viridis')
    
    ax1.contour(X, Y, losses.T, levels=15, colors='white', alpha=0.3, linewidths=0.5)
    ax1.plot(center_w00, center_w01, 'r*', markersize=20, label='Model parameters', markeredgecolor='white', markeredgewidth=1.5)
    ax1.set_xlabel('fc1.weight[0, 0]', fontsize=12)
    ax1.set_ylabel('fc1.weight[0, 1]', fontsize=12)
    ax1.set_title('Complete Loss Surface (Contour)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(contour, ax=ax1)
    cbar1.set_label('Loss (BCE)', fontsize=11)
    
    # 3D surface plot
    ax2 = fig.add_subplot(132, projection='3d')
    if loss_max / loss_min > 100:
        surf = ax2.plot_surface(X, Y, losses.T, cmap='viridis', 
                                norm='log', alpha=0.9, antialiased=True, 
                                edgecolor='none', rcount=50, ccount=50)
    else:
        surf = ax2.plot_surface(X, Y, losses.T, cmap='viridis', 
                                alpha=0.9, antialiased=True,
                                edgecolor='none', rcount=50, ccount=50)
    
    # Find the center point loss
    center_i = grid_size // 2
    center_j = grid_size // 2
    ax2.plot([center_w00], [center_w01], [losses[center_i, center_j]], 
             'r*', markersize=20, label='Model parameters',
             markeredgecolor='white', markeredgewidth=1.5)
    ax2.set_xlabel('fc1.weight[0, 0]', fontsize=11)
    ax2.set_ylabel('fc1.weight[0, 1]', fontsize=11)
    ax2.set_zlabel('Loss (BCE)', fontsize=11)
    ax2.set_title('Complete Loss Surface (3D)', fontsize=14, fontweight='bold')
    ax2.view_init(elev=25, azim=45)
    cbar2 = plt.colorbar(surf, ax=ax2, shrink=0.6, pad=0.1)
    cbar2.set_label('Loss (BCE)', fontsize=11)
    
    # Heatmap with cross-sections
    ax3 = plt.subplot(133)
    im = ax3.imshow(losses.T, extent=[w00_values[0], w00_values[-1], w01_values[0], w01_values[-1]],
                    origin='lower', cmap='viridis', aspect='auto',
                    norm='log' if loss_max / loss_min > 100 else None)
    ax3.plot(center_w00, center_w01, 'r*', markersize=20, label='Model parameters',
             markeredgecolor='white', markeredgewidth=1.5)
    
    # Add cross-sections through the center point
    ax3.axvline(center_w00, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax3.axhline(center_w01, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    ax3.set_xlabel('fc1.weight[0, 0]', fontsize=12)
    ax3.set_ylabel('fc1.weight[0, 1]', fontsize=12)
    ax3.set_title('Complete Loss Surface (Heatmap)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    cbar3 = plt.colorbar(im, ax=ax3)
    cbar3.set_label('Loss (BCE)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComplete loss surface plot saved to: {save_path}")
    plt.show()
    
    # Print statistics
    print(f"\nLoss Surface Statistics:")
    print(f"  Minimum loss: {loss_min:.6f} at w00={w00_values[np.unravel_index(losses.argmin(), losses.shape)[0]]:.4f}, w01={w01_values[np.unravel_index(losses.argmin(), losses.shape)[1]]:.4f}")
    print(f"  Maximum loss: {loss_max:.6f}")
    print(f"  Loss at center: {losses[center_i, center_j]:.6f}")
    print(f"  Loss range: {loss_max - loss_min:.6f}")
    print(f"  Loss ratio (max/min): {loss_max / loss_min:.2f}")


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
    
    # Plot COMPLETE loss surface (actual weight space) for Muon model
    print("\n\nVisualizing COMPLETE loss surface for trained Muon model...")
    print("(varying fc1.weight[0,0] and fc1.weight[0,1] while keeping all other params fixed)")
    plot_complete_loss_surface(
        model=results['muon'][0]['model'], 
        grid_size=100, 
        extent=3.0,
        save_path='complete_loss_surface_muon.png'
    )
    
    # Plot COMPLETE loss surface (actual weight space) for AdamW model
    print("\n\nVisualizing COMPLETE loss surface for trained AdamW model...")
    print("(varying fc1.weight[0,0] and fc1.weight[0,1] while keeping all other params fixed)")
    plot_complete_loss_surface(
        model=results['adamw'][0]['model'], 
        grid_size=100, 
        extent=3.0,
        save_path='complete_loss_surface_adamw.png'
    )
    
    # Also plot random projection loss surfaces for comparison
    print("\n\n" + "="*70)
    print("BONUS: Random projection loss surfaces (for comparison)")
    print("="*70)
    
    print("\nVisualizing random projection loss surface for Muon model...")
    plot_loss_surface(
        model=results['muon'][0]['model'], 
        grid_size=50, 
        extent=1.5,
        save_path='random_loss_surface_muon.png'
    )
    
    print("\nVisualizing random projection loss surface for AdamW model...")
    plot_loss_surface(
        model=results['adamw'][0]['model'], 
        grid_size=50, 
        extent=1.5,
        save_path='random_loss_surface_adamw.png'
    )
