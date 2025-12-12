import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_training_curves(history_path, save_dir):
    """
    Plot training and validation loss/accuracy curves
    """
    with open(history_path, 'r') as f:
        data = json.load(f)
    
    phase1 = data['phase1']
    phase2 = data['phase2']
    class_names = data.get('class_names', [])
    
    train_loss = phase1['train_loss'] + phase2['train_loss']
    train_acc = phase1['train_acc'] + phase2['train_acc']
    val_loss = phase1['val_loss'] + phase2['val_loss']
    val_acc = phase1['val_acc'] + phase2['val_acc']
    
    phase1_epochs = len(phase1['train_loss'])
    phase2_epochs = len(phase2['train_loss'])
    total_epochs = phase1_epochs + phase2_epochs
    
    epochs = np.arange(1, total_epochs + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # plot 1: Loss
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2, marker='o')
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2, marker='s')
    ax1.axvline(x=phase1_epochs, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Phase 1→2')
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.set_title('Training and Validation Loss', fontsize=16, pad=15)
    ax1.legend(fontsize=12)
    ax1.grid(alpha=0.3)
    
    ax1.text(phase1_epochs/2, max(train_loss)*0.95, 'Phase 1\n(Frozen)', 
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.text(phase1_epochs + phase2_epochs/2, max(train_loss)*0.95, 'Phase 2\n(Fine-tune)', 
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # plot 2: Accuracy
    ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2, marker='o')
    ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2, marker='s')
    ax2.axvline(x=phase1_epochs, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Phase 1→2')
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Accuracy', fontsize=14)
    ax2.set_title('Training and Validation Accuracy', fontsize=16, pad=15)
    ax2.legend(fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    ax2.text(phase1_epochs/2, 0.05, 'Phase 1\n(Frozen)', 
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.text(phase1_epochs + phase2_epochs/2, 0.05, 'Phase 2\n(Fine-tune)', 
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    save_path = f"{save_dir}/training_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training curves to {save_path}")
    
    print(f"Training Summary:")
    print(f"  Phase 1 (Frozen Backbone):")
    print(f"    Final Train Loss: {phase1['train_loss'][-1]:.4f}, Accuracy: {phase1['train_acc'][-1]:.4f}")
    print(f"    Final Val Loss:   {phase1['val_loss'][-1]:.4f}, Accuracy: {phase1['val_acc'][-1]:.4f}")
    print(f"  Phase 2 (Fine-tuning):")
    print(f"    Final Train Loss: {phase2['train_loss'][-1]:.4f}, Accuracy: {phase2['train_acc'][-1]:.4f}")
    print(f"    Final Val Loss:   {phase2['val_loss'][-1]:.4f}, Accuracy: {phase2['val_acc'][-1]:.4f}")
    print(f"  Best Validation Accuracy: {max(val_acc):.4f} (Epoch {np.argmax(val_acc)+1})")


def plot_loss_landscape_concept(save_dir):
    """
    Create a conceptual illustration of loss landscape during training
    This is a stylized representation, not actual loss landscape
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a mesh
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # create a loss-like surface (double-well potential)
    Z = (X**2 + Y**2) + 5*np.sin(X) + 5*np.sin(Y) + 0.1*X*Y
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')
    
    # simulate training trajectory (gradient descent)
    trajectory_x = np.linspace(-2, 0.5, 50)
    trajectory_y = np.linspace(-2, 0.3, 50)
    trajectory_z = [(x**2 + y**2) + 5*np.sin(x) + 5*np.sin(y) + 0.1*x*y 
                    for x, y in zip(trajectory_x, trajectory_y)]
    
    ax.plot(trajectory_x, trajectory_y, trajectory_z, 'r-', linewidth=3, label='Training Path')
    ax.scatter(trajectory_x[0], trajectory_y[0], trajectory_z[0], 
               c='green', s=200, marker='o', label='Start', edgecolors='black', linewidth=2)
    ax.scatter(trajectory_x[-1], trajectory_y[-1], trajectory_z[-1], 
               c='red', s=200, marker='*', label='Final Model', edgecolors='black', linewidth=2)
    
    ax.set_xlabel('Parameter Dimension 1', fontsize=12, labelpad=10)
    ax.set_ylabel('Parameter Dimension 2', fontsize=12, labelpad=10)
    ax.set_zlabel('Loss', fontsize=12, labelpad=10)
    ax.set_title('Conceptual Loss Landscape During Training', fontsize=14, pad=20)
    ax.legend(fontsize=10)
    ax.view_init(elev=25, azim=45)
    
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    save_path = f"{save_dir}/loss_landscape_concept.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved conceptual loss landscape to {save_path}")


def plot_phase_comparison(history_path, save_dir):
    """
    Compare Phase 1 vs Phase 2 performance
    """
    with open(history_path, 'r') as f:
        data = json.load(f)
    
    phase1 = data['phase1']
    phase2 = data['phase2']
    
    metrics = {
        'Train Loss': [phase1['train_loss'][-1], phase2['train_loss'][-1]],
        'Val Loss': [phase1['val_loss'][-1], phase2['val_loss'][-1]],
        'Train Acc': [phase1['train_acc'][-1], phase2['train_acc'][-1]],
        'Val Acc': [phase1['val_acc'][-1], phase2['val_acc'][-1]]
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[idx]
        phases = ['Phase 1\n(Frozen)', 'Phase 2\n(Fine-tuned)']
        colors = ['skyblue', 'lightcoral']
        bars = ax.bar(phases, values, color=colors, edgecolor='black', linewidth=2, alpha=0.7)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        if 'Loss' in metric_name:
            improvement = ((values[0] - values[1]) / values[0]) * 100
            label = f"↓ {improvement:.1f}% (better)" if improvement > 0 else f"↑ {-improvement:.1f}% (worse)"
        else:
            improvement = ((values[1] - values[0]) / values[0]) * 100
            label = f"↑ {improvement:.1f}% (better)" if improvement > 0 else f"↓ {-improvement:.1f}% (worse)"
        
        ax.text(0.5, 0.95, label, transform=ax.transAxes,
               ha='center', va='top', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name} Comparison', fontsize=13, pad=10)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Phase 1 vs Phase 2 Performance Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    
    save_path = f"{save_dir}/phase_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved phase comparison to {save_path}")


def main():
    """Main function to generate all training visualizations"""
    history_path = "checkpoints/training_history.json"
    save_dir = "training_visualizations"
    
    Path(save_dir).mkdir(exist_ok=True)
    
    print("Generating Training Visualizations\n")
    
    if not Path(history_path).exists():
        print(f"Error: Training history not found at {history_path}")
        print("   Please run train.py first to generate training history.")
        return
    
    print("1. Plotting training curves...")
    plot_training_curves(history_path, save_dir)
    
    print("\n2. Plotting phase comparison...")
    plot_phase_comparison(history_path, save_dir)
    
    print("\n3. Creating conceptual loss landscape...")
    plot_loss_landscape_concept(save_dir)
    
    print(f"All visualizations complete!")
    print(f"Saved to: {save_dir}/")


if __name__ == "__main__":
    main()