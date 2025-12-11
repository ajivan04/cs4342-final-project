"""
Evaluation Script for Outfit Classification Model
Generates detailed metrics, confusion matrix, and example predictions
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)
from pathlib import Path
import json
from PIL import Image


class Config:
    DATA_DIR = "data/clean_data"
    NUM_CLASSES = 4
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    CHECKPOINT_PATH = "checkpoints/final_model.pth"
    RESULTS_DIR = "evaluation_results"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_test_loader(config):
    """Create test dataloader"""
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    test_dataset = datasets.ImageFolder(
        f"{config.DATA_DIR}/test",
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    return test_loader, test_dataset


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


def get_predictions(model, loader, device):
    """Get all predictions and labels"""
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion matrix to {save_path}")


def plot_normalized_confusion_matrix(cm, class_names, save_path):
    """Plot normalized confusion matrix (percentages)"""
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized, annot=True, fmt='.2%', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage'}
    )
    plt.title('Normalized Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved normalized confusion matrix to {save_path}")


def plot_per_class_metrics(precision, recall, f1, class_names, save_path):
    """Plot per-class metrics"""
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved per-class metrics to {save_path}")


def find_misclassified_examples(test_dataset, labels, preds, probs, class_names, num_examples=5):
    """Find examples of misclassified images"""
    misclassified = []
    
    for idx, (true_label, pred_label) in enumerate(zip(labels, preds)):
        if true_label != pred_label:
            img_path = test_dataset.samples[idx][0]
            misclassified.append({
                'index': idx,
                'path': img_path,
                'true_label': class_names[true_label],
                'pred_label': class_names[pred_label],
                'confidence': float(probs[idx][pred_label]),
                'true_confidence': float(probs[idx][true_label])
            })
    
    # Sort by confidence (most confident mistakes first)
    misclassified.sort(key=lambda x: x['confidence'], reverse=True)
    
    return misclassified[:num_examples]


def calculate_top_k_accuracy(labels, probs, k=2):
    """Calculate top-k accuracy"""
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    correct = sum([labels[i] in top_k_preds[i] for i in range(len(labels))])
    return correct / len(labels)


def evaluate_model(config):
    """Main evaluation function"""
    print("🔍 Starting Model Evaluation")
    print(f"Device: {config.DEVICE}")
    
    # Create results directory
    Path(config.RESULTS_DIR).mkdir(exist_ok=True)
    
    # Load data
    print("\n📦 Loading test data...")
    test_loader, test_dataset = get_test_loader(config)
    class_names = test_dataset.classes
    print(f"Test set size: {len(test_dataset)} images")
    print(f"Classes: {class_names}")
    
    # Load model
    print(f"\n🏗️  Loading model from {config.CHECKPOINT_PATH}...")
    model, checkpoint = load_model(config.CHECKPOINT_PATH, config.NUM_CLASSES, config.DEVICE)
    print("Model loaded successfully!")
    
    # Get predictions
    print("\n🎯 Generating predictions...")
    labels, preds, probs = get_predictions(model, test_loader, config.DEVICE)
    
    # Calculate metrics
    print("\n📊 Calculating metrics...")
    
    # Overall accuracy
    accuracy = accuracy_score(labels, preds)
    print(f"\n{'='*60}")
    print(f"OVERALL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*60}")
    
    # Top-2 accuracy
    top2_acc = calculate_top_k_accuracy(labels, probs, k=2)
    print(f"Top-2 Accuracy: {top2_acc:.4f} ({top2_acc*100:.2f}%)")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, labels=range(config.NUM_CLASSES)
    )
    
    print(f"\n{'='*60}")
    print("PER-CLASS METRICS")
    print(f"{'='*60}")
    print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 70)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<25} {precision[i]:>10.4f} {recall[i]:>10.4f} "
              f"{f1[i]:>10.4f} {support[i]:>10.0f}")
    
    # Macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    print("-" * 70)
    print(f"{'Macro Average':<25} {macro_precision:>10.4f} {macro_recall:>10.4f} {macro_f1:>10.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print(f"\n{'='*60}")
    print("CONFUSION MATRIX")
    print(f"{'='*60}")
    print(cm)
    
    # Save classification report
    report = classification_report(
        labels, preds,
        target_names=class_names,
        digits=4
    )
    print(f"\n{report}")
    
    # Save results to file
    results = {
        'overall_accuracy': float(accuracy),
        'top2_accuracy': float(top2_acc),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'per_class': {
            class_names[i]: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
            for i in range(config.NUM_CLASSES)
        },
        'confusion_matrix': cm.tolist()
    }
    
    results_path = f"{config.RESULTS_DIR}/evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {results_path}")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    
    plot_confusion_matrix(
        cm, class_names,
        f"{config.RESULTS_DIR}/confusion_matrix.png"
    )
    
    plot_normalized_confusion_matrix(
        cm, class_names,
        f"{config.RESULTS_DIR}/confusion_matrix_normalized.png"
    )
    
    plot_per_class_metrics(
        precision, recall, f1, class_names,
        f"{config.RESULTS_DIR}/per_class_metrics.png"
    )
    
    # Find misclassified examples
    print("\n🔍 Finding misclassified examples...")
    misclassified = find_misclassified_examples(
        test_dataset, labels, preds, probs, class_names, num_examples=10
    )
    
    misclassified_path = f"{config.RESULTS_DIR}/misclassified_examples.json"
    with open(misclassified_path, 'w') as f:
        json.dump(misclassified, f, indent=2)
    
    print(f"\nTop 10 misclassified examples (most confident mistakes):")
    for i, example in enumerate(misclassified[:5], 1):
        print(f"\n{i}. {Path(example['path']).name}")
        print(f"   True: {example['true_label']}")
        print(f"   Predicted: {example['pred_label']} (confidence: {example['confidence']:.4f})")
    
    print(f"\n✅ Evaluation complete!")
    print(f"All results saved to: {config.RESULTS_DIR}/")


if __name__ == "__main__":
    config = Config()
    evaluate_model(config)