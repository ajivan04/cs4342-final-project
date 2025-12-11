import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict
import random

# Configuration
RAW_DATA_DIR = "data/raw_data"  # Your original images
CLEAN_DATA_DIR = "data/clean_data"
CATEGORIES = ["formal_professional", "party_social", "gym_athletic", "casual_everyday"]
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
MIN_IMAGE_SIZE = 224  # Minimum dimension
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def is_valid_image(image_path):
    """Check if image is valid and meets minimum size requirements"""
    try:
        with Image.open(image_path) as img:
            # Check if image can be loaded
            img.verify()
        
        # Reopen for size check (verify() closes the file)
        with Image.open(image_path) as img:
            width, height = img.size
            
            # Check minimum size
            if min(width, height) < MIN_IMAGE_SIZE:
                print(f"⚠️  Image too small: {image_path} ({width}x{height})")
                return False
            
            # Check if image is corrupted
            img.load()
            
            return True
    except Exception as e:
        print(f"❌ Invalid image {image_path}: {e}")
        return False


def resize_and_save(src_path, dst_path, max_size=1024):
    """Resize image if too large while maintaining aspect ratio"""
    try:
        with Image.open(src_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if too large
            width, height = img.size
            if max(width, height) > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                
                img = img.resize((new_width, new_height), Image.LANCZOS)
                print(f"  Resized: {width}x{height} → {new_width}x{new_height}")
            
            # Save
            img.save(dst_path, 'JPEG', quality=95)
            return True
    except Exception as e:
        print(f"❌ Error processing {src_path}: {e}")
        return False


def split_data_random(image_files, train_ratio, val_ratio, test_ratio):
    """
    Split data randomly into train/val/test sets
    """
    # Shuffle all images
    image_files = list(image_files)
    random.shuffle(image_files)
    
    # Calculate split indices
    n_images = len(image_files)
    n_train = int(n_images * train_ratio)
    n_val = int(n_images * val_ratio)
    
    # Split into train/val/test
    train_images = image_files[:n_train]
    val_images = image_files[n_train:n_train + n_val]
    test_images = image_files[n_train + n_val:]
    
    return train_images, val_images, test_images


def clean_and_organize_data():
    """Main function to clean and organize data"""
    print("🧹 Starting data cleaning and organization...\n")
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        for category in CATEGORIES:
            Path(f"{CLEAN_DATA_DIR}/{split}/{category}").mkdir(parents=True, exist_ok=True)
    
    stats = {
        'total': 0,
        'valid': 0,
        'invalid': 0,
        'train': 0,
        'val': 0,
        'test': 0
    }
    
    category_stats = {cat: {'train': 0, 'val': 0, 'test': 0} for cat in CATEGORIES}
    
    # Process each category
    for category in CATEGORIES:
        print(f"\n📁 Processing category: {category}")
        category_path = Path(RAW_DATA_DIR) / category
        
        if not category_path.exists():
            print(f"⚠️  Warning: {category_path} does not exist. Skipping...")
            continue
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(category_path.glob(ext)))
        
        print(f"  Found {len(image_files)} images")
        stats['total'] += len(image_files)
        
        # Validate images
        valid_images = []
        for img_path in image_files:
            if is_valid_image(img_path):
                valid_images.append(img_path)
                stats['valid'] += 1
            else:
                stats['invalid'] += 1
        
        print(f"  Valid images: {len(valid_images)}")
        
        if len(valid_images) == 0:
            print(f"  ⚠️  No valid images found for {category}")
            continue
        
        # Split into train/val/test
        train_imgs, val_imgs, test_imgs = split_data_random(
            valid_images, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
        )
        
        print(f"  Split: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")
        
        # Copy images to appropriate directories
        for split, images in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
            for i, img_path in enumerate(images):
                dst_path = Path(f"{CLEAN_DATA_DIR}/{split}/{category}/{category}_{split}_{i:04d}.jpg")
                if resize_and_save(img_path, dst_path):
                    category_stats[category][split] += 1
                    stats[split] += 1
    
    # Print summary
    print("\n" + "="*60)
    print("📊 DATA CLEANING SUMMARY")
    print("="*60)
    print(f"Total images processed: {stats['total']}")
    print(f"Valid images: {stats['valid']}")
    print(f"Invalid/rejected images: {stats['invalid']}")
    print(f"\nDataset split:")
    print(f"  Train: {stats['train']} ({stats['train']/stats['valid']*100:.1f}%)")
    print(f"  Val:   {stats['val']} ({stats['val']/stats['valid']*100:.1f}%)")
    print(f"  Test:  {stats['test']} ({stats['test']/stats['valid']*100:.1f}%)")
    
    print(f"\n📈 Per-category breakdown:")
    for category in CATEGORIES:
        total = sum(category_stats[category].values())
        if total > 0:
            print(f"\n  {category}:")
            print(f"    Train: {category_stats[category]['train']}")
            print(f"    Val:   {category_stats[category]['val']}")
            print(f"    Test:  {category_stats[category]['test']}")
            print(f"    Total: {total}")
    
    print("\n✅ Data cleaning complete!")
    print(f"Clean data saved to: {CLEAN_DATA_DIR}/")


if __name__ == "__main__":
    clean_and_organize_data()