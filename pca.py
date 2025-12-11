import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from pathlib import Path
import json


class Config:
    DATA_DIR = "data/clean_data"
    NUM_CLASSES = 4
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    CHECKPOINT_PATH = "checkpoints/final_model.pth"
    RESULTS_DIR = "pca_visualizations"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloaders(config):
    """Create dataloaders for train and test sets"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    train_dataset = datasets.ImageFolder(f"{config.DATA_DIR}/train", transform=transform)
    test_dataset = datasets.ImageFolder(f"{config.DATA_DIR}/test", transform=transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS
    )
    
    return train_loader, test_loader, train_dataset.classes


def load_model(checkpoint_path, num_classes, device):
    """Load trained model from checkpoint"""
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


class FeatureExtractor(nn.Module):
    """Extract features from the layer before final classification"""
    def __init__(self, model):
        super().__init__()
        # Remove the final fc layer
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.fc = model.fc
        
    def forward(self, x):
        features = self.features(x)
        features = torch.flatten(features, 1)
        logits = self.fc(features)
        return features, logits


def extract_features_and_predictions(model, loader, device, max_samples=1000):
    """
    Extract features and predictions from the model
    Returns features (input representations), predictions, labels, and probabilities
    """
    feature_extractor = FeatureExtractor(model)
    feature_extractor.eval()
    
    all_features = []
    all_predictions = []
    all_labels = []
    all_probs = []
    
    sample_count = 0
    
    with torch.no_grad():
        for images, labels in loader:
            if sample_count >= max_samples:
                break
                
            images = images.to(device)
            features, logits = feature_extractor(images)
            probs = torch.softmax(logits, dim=1)
            _, preds = logits.max(1)
            
            all_features.append(features.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())
            
            sample_count += images.size(0)
    
    all_features = np.vstack(all_features)[:max_samples]
    all_predictions = np.concatenate(all_predictions)[:max_samples]
    all_labels = np.concatenate(all_labels)[:max_samples]
    all_probs = np.vstack(all_probs)[:max_samples]
    
    return all_features, all_predictions, all_labels, all_probs


def plot_pca_predictions_3d(features, predictions, labels, class_names, save_path, title):
    """
    Create 3D PCA visualization of g(x) where:
    - X, Y axes: First 2 principal components of input x
    - Z axis: Predicted class ŷ
    """
    # Apply PCA to reduce features to 2D
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color map for classes
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    
    # Plot each class
    for class_idx in range(len(class_names)):
        mask = predictions == class_idx
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            predictions[mask],
            c=[colors[class_idx]],
            label=class_names[class_idx],
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
    
    # Labels and title
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12, labelpad=10)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12, labelpad=10)
    ax.set_zlabel('Predicted Class ŷ', fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, pad=20)
    
    # Set z-axis to show class indices
    ax.set_zticks(range(len(class_names)))
    ax.set_zticklabels(range(len(class_names)))
    
    # Legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)
    
    # Viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved 3D prediction visualization to {save_path}")
    
    # Return PCA object and explained variance
    return pca


def plot_pca_with_confidence(features, predictions, probs, labels, class_names, save_path):
    """
    Create 3D PCA visualization where:
    - X, Y axes: First 2 principal components
    - Z axis: Prediction confidence (max probability)
    """
    # Apply PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    # Get maximum probability (confidence)
    confidence = np.max(probs, axis=1)
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by correctness
    correct = predictions == labels
    
    # Plot correct predictions
    mask_correct = correct
    scatter_correct = ax.scatter(
        features_2d[mask_correct, 0],
        features_2d[mask_correct, 1],
        confidence[mask_correct],
        c='green',
        label='Correct',
        alpha=0.6,
        s=50,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Plot incorrect predictions
    mask_incorrect = ~correct
    if mask_incorrect.sum() > 0:
        scatter_incorrect = ax.scatter(
            features_2d[mask_incorrect, 0],
            features_2d[mask_incorrect, 1],
            confidence[mask_incorrect],
            c='red',
            label='Incorrect',
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
    
    # Labels
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12, labelpad=10)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12, labelpad=10)
    ax.set_zlabel('Prediction Confidence', fontsize=12, labelpad=10)
    ax.set_title('PCA Visualization: Input Space vs Confidence', fontsize=14, pad=20)
    
    ax.legend(fontsize=10)
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved confidence visualization to {save_path}")


def plot_pca_2d_by_class(features, predictions, labels, class_names, save_path):
    """
    Create 2D PCA visualization colored by predicted class
    """
    # Apply PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    
    # Plot by predicted class
    for class_idx in range(len(class_names)):
        mask = predictions == class_idx
        ax1.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[class_idx]],
            label=class_names[class_idx],
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax1.set_title('PCA by Predicted Class', fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    
    # Plot by true class
    for class_idx in range(len(class_names)):
        mask = labels == class_idx
        ax2.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[class_idx]],
            label=class_names[class_idx],
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
    
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax2.set_title('PCA by True Class', fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved 2D PCA comparison to {save_path}")


def plot_class_separation(features, labels, class_names, save_path):
    """
    Visualize class separation in PCA space with decision boundaries
    """
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    
    # Plot each class with larger markers
    for class_idx in range(len(class_names)):
        mask = labels == class_idx
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[class_idx]],
            label=class_names[class_idx],
            alpha=0.6,
            s=100,
            edgecolors='black',
            linewidth=1
        )
        
        # Add class centroid
        centroid = features_2d[mask].mean(axis=0)
        ax.scatter(
            centroid[0], centroid[1],
            c=[colors[class_idx]],
            marker='*',
            s=500,
            edgecolors='black',
            linewidth=2,
            zorder=10
        )
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=14)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=14)
    ax.set_title('Class Separation in PCA Space\n(Stars indicate class centroids)', fontsize=16)
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved class separation plot to {save_path}")


def analyze_pca_components(pca, save_path):
    """
    Analyze and save PCA component information
    """
    info = {
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
        'n_components': pca.n_components_
    }
    
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"  Saved PCA analysis to {save_path}")
    
    # Print summary
    print(f"\n  PCA Summary:")
    print(f"    PC1 explains {pca.explained_variance_ratio_[0]:.2%} of variance")
    print(f"    PC2 explains {pca.explained_variance_ratio_[1]:.2%} of variance")
    print(f"    Total: {sum(pca.explained_variance_ratio_):.2%} of variance captured")


def visualize_pca_predictions():
    """Main visualization function"""
    config = Config()
    
    print("🎨 Starting PCA Visualization")
    print(f"Device: {config.DEVICE}")
    
    # Create output directory
    Path(config.RESULTS_DIR).mkdir(exist_ok=True)
    
    # Load data
    print("\n📦 Loading data...")
    train_loader, test_loader, class_names = get_dataloaders(config)
    print(f"Classes: {class_names}")
    
    # Load model
    print(f"\n🏗️  Loading model from {config.CHECKPOINT_PATH}...")
    model, checkpoint = load_model(config.CHECKPOINT_PATH, config.NUM_CLASSES, config.DEVICE)
    print("Model loaded successfully!")
    
    # Extract features and predictions from test set
    print("\n📊 Extracting features and predictions from test set...")
    features, predictions, labels, probs = extract_features_and_predictions(
        model, test_loader, config.DEVICE, max_samples=1000
    )
    print(f"Extracted {len(features)} samples")
    print(f"Feature dimension: {features.shape[1]}")
    
    # Generate visualizations
    print("\n🎨 Generating PCA visualizations...")
    
    # 1. Main 3D plot: g(x) with PCA dimensions and predictions
    print("\n1. Creating 3D prediction visualization (g(x))...")
    pca = plot_pca_predictions_3d(
        features, predictions, labels, class_names,
        f"{config.RESULTS_DIR}/pca_predictions_3d.png",
        "3D Visualization: g(x) - Input Features (PCA) vs Predictions"
    )
    
    # 2. 3D plot with confidence
    print("\n2. Creating confidence visualization...")
    plot_pca_with_confidence(
        features, predictions, probs, labels, class_names,
        f"{config.RESULTS_DIR}/pca_confidence_3d.png"
    )
    
    # 3. 2D comparison plots
    print("\n3. Creating 2D comparison plots...")
    plot_pca_2d_by_class(
        features, predictions, labels, class_names,
        f"{config.RESULTS_DIR}/pca_comparison_2d.png"
    )
    
    # 4. Class separation visualization
    print("\n4. Creating class separation plot...")
    plot_class_separation(
        features, labels, class_names,
        f"{config.RESULTS_DIR}/pca_class_separation.png"
    )
    
    # 5. Analyze PCA components
    print("\n5. Analyzing PCA components...")
    analyze_pca_components(pca, f"{config.RESULTS_DIR}/pca_analysis.json")
    
    # Extract features from training set for comparison
    print("\n📊 Extracting features from training set...")
    train_features, train_predictions, train_labels, train_probs = extract_features_and_predictions(
        model, train_loader, config.DEVICE, max_samples=1000
    )
    
    # 6. Training set visualization
    print("\n6. Creating training set visualization...")
    plot_pca_predictions_3d(
        train_features, train_predictions, train_labels, class_names,
        f"{config.RESULTS_DIR}/pca_predictions_3d_train.png",
        "3D Visualization: g(x) - Training Set"
    )
    
    print(f"\n✅ PCA visualization complete!")
    print(f"All visualizations saved to: {config.RESULTS_DIR}/")
    print(f"\nGenerated files:")
    print(f"  - pca_predictions_3d.png (Main 3D plot for assignment)")
    print(f"  - pca_confidence_3d.png")
    print(f"  - pca_comparison_2d.png")
    print(f"  - pca_class_separation.png")
    print(f"  - pca_predictions_3d_train.png")
    print(f"  - pca_analysis.json")


if __name__ == "__main__":
    visualize_pca_predictions()