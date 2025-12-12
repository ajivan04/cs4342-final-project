import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from collections import Counter


# Configuration
class Config:
    DATA_DIR = "data/clean_data"
    NUM_CLASSES = 4
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # phase 1
    PHASE1_EPOCHS = 10
    PHASE1_LR = 1e-3
    PHASE1_WEIGHT_DECAY = 1e-4
    
    # phase 2
    PHASE2_EPOCHS = 10
    PHASE2_LR = 1e-4
    PHASE2_WEIGHT_DECAY = 1e-4
    
    MODEL_NAME = "resnet18"
    CHECKPOINT_DIR = "checkpoints"
    RANDOM_SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_transforms():
    """Get train and validation transforms"""
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    return train_transform, val_test_transform


def get_dataloaders(config):
    """Create dataloaders for train, val, and test sets"""
    train_transform, val_test_transform = get_transforms()
    
    # Load datasets
    train_dataset = datasets.ImageFolder(
        f"{config.DATA_DIR}/train",
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        f"{config.DATA_DIR}/val",
        transform=val_test_transform
    )
    test_dataset = datasets.ImageFolder(
        f"{config.DATA_DIR}/test",
        transform=val_test_transform
    )
    
    class_counts = Counter([label for _, label in train_dataset.samples])
    print("Training set class distribution:")
    for idx, count in sorted(class_counts.items()):
        print(f"  {train_dataset.classes[idx]}: {count} images")
    
    total = sum(class_counts.values())
    class_weights = torch.tensor(
        [total / class_counts[i] for i in range(config.NUM_CLASSES)],
        dtype=torch.float32
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_weights, train_dataset.classes


def create_model(config):
    """Create ResNet-18 model with custom final layer"""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, config.NUM_CLASSES)
    
    return model


def freeze_backbone(model):
    """Freeze all layers except the final fc layer"""
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.fc.parameters():
        param.requires_grad = True
    
    print("🔒 Backbone frozen. Only training final layer.")


def unfreeze_last_blocks(model, num_blocks=2):
    """Unfreeze last few residual blocks for fine-tuning"""
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.fc.parameters():
        param.requires_grad = True
    
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    if num_blocks >= 2:
        for param in model.layer3.parameters():
            param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"🔓 Unfroze last {num_blocks} blocks. Training {trainable:,} / {total:,} parameters ({trainable/total*100:.1f}%)")


def train_epoch(model, loader, criterion, optimizer, device, config):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def train_phase(model, train_loader, val_loader, config, phase_num, num_epochs, lr, optimizer=None):
    """Train model for a specific phase"""
    print(f"PHASE {phase_num}: {'Frozen Backbone' if phase_num == 1 else 'Fine-tuning'}")
    
    _, _, _, class_weights, _ = get_dataloaders(config)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(config.DEVICE))
    
    if optimizer is None:
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=config.PHASE1_WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # training loop
    best_val_acc = 0.0
    best_model_path = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, config
        )
        
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = f"{config.CHECKPOINT_DIR}/best_phase{phase_num}_epoch{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, best_model_path)
            print(f"Saved best model: {best_model_path}")
    
    print(f"Phase {phase_num} Best Val Accuracy: {best_val_acc:.4f}")
    
    return model, best_model_path, history, best_val_acc


def main():
    """Main training function"""
    config = Config()
    set_seed(config.RANDOM_SEED)
    
    Path(config.CHECKPOINT_DIR).mkdir(exist_ok=True)
    
    print(f"Starting Outfit Classification Training")
    print(f"Device: {config.DEVICE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    
    # load data
    print("Loading data...")
    train_loader, val_loader, test_loader, class_weights, class_names = get_dataloaders(config)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset)} images")
    print(f"  Val:   {len(val_loader.dataset)} images")
    print(f"  Test:  {len(test_loader.dataset)} images")
    print(f"\nClasses: {class_names}")
    
    # create model
    print("Creating model...")
    model = create_model(config)
    model = model.to(config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # phase 1
    freeze_backbone(model)
    model, phase1_path, phase1_history, phase1_acc = train_phase(
        model, train_loader, val_loader, config,
        phase_num=1,
        num_epochs=config.PHASE1_EPOCHS,
        lr=config.PHASE1_LR
    )
    
    # phase 2
    unfreeze_last_blocks(model, num_blocks=2)
    model, phase2_path, phase2_history, phase2_acc = train_phase(
        model, train_loader, val_loader, config,
        phase_num=2,
        num_epochs=config.PHASE2_EPOCHS,
        lr=config.PHASE2_LR
    )
    
    final_path = f"{config.CHECKPOINT_DIR}/final_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'config': vars(config),
        'phase1_acc': phase1_acc,
        'phase2_acc': phase2_acc,
    }, final_path)
    
    print(f"Training complete!")
    print(f"Final model saved to: {final_path}")
    print(f"Phase 1 best accuracy: {phase1_acc:.4f}")
    print(f"Phase 2 best accuracy: {phase2_acc:.4f}")
    
    history_path = f"{config.CHECKPOINT_DIR}/training_history.json"
    with open(history_path, 'w') as f:
        json.dump({
            'phase1': phase1_history,
            'phase2': phase2_history,
            'class_names': class_names
        }, f, indent=2)
    
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()