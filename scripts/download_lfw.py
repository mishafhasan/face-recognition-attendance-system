"""
LFW Dataset Download and Preprocessing Script

Downloads the Labeled Faces in the Wild (LFW) dataset and prepares it for
training a face embedding model with triplet loss.

Usage:
    python download_lfw.py
    python download_lfw.py --min-faces 20  # Only people with 20+ images
"""
import os
import sys
import argparse
import pickle
from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_lfw_people, fetch_lfw_pairs
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LFW_DIR = DATA_DIR / "lfw"
PROCESSED_DIR = LFW_DIR / "processed"


def download_lfw_people(min_faces: int = 20) -> dict:
    """
    Download LFW dataset using sklearn.
    
    Args:
        min_faces: Minimum number of faces per person to include
        
    Returns:
        Dictionary with images, labels, and target names
    """
    print(f"\n📥 Downloading LFW dataset (min {min_faces} faces per person)...")
    print("   This may take a few minutes on first run...")
    
    lfw = fetch_lfw_people(
        min_faces_per_person=min_faces,
        resize=1.0,  # Keep original size
        color=True,
        download_if_missing=True
    )
    
    print(f"✅ Downloaded successfully!")
    print(f"   Total images: {len(lfw.images)}")
    print(f"   Total people: {len(lfw.target_names)}")
    print(f"   Image shape: {lfw.images[0].shape}")
    
    return {
        "images": lfw.images,
        "labels": lfw.target,
        "names": lfw.target_names
    }


def download_lfw_pairs() -> dict:
    """
    Download LFW verification pairs for evaluation.
    
    Returns:
        Dictionary with pairs and labels (same/different)
    """
    print("\n📥 Downloading LFW verification pairs...")
    
    pairs = fetch_lfw_pairs(subset='test', color=True)
    
    print(f"✅ Downloaded {len(pairs.pairs)} pairs for evaluation")
    
    return {
        "pairs": pairs.pairs,
        "labels": pairs.target  # 1 = same person, 0 = different
    }


def preprocess_images(images: np.ndarray, target_size: int = 160) -> np.ndarray:
    """
    Preprocess images for face embedding model.
    
    Args:
        images: Array of face images
        target_size: Output image size (default 160 for FaceNet)
        
    Returns:
        Preprocessed images normalized to [-1, 1]
    """
    print(f"\n🔄 Preprocessing {len(images)} images to {target_size}x{target_size}...")
    
    processed = []
    for img in tqdm(images, desc="Processing"):
        # Convert to uint8 if float
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        # Resize to target size
        resized = cv2.resize(img, (target_size, target_size))
        
        # Normalize to [-1, 1] (standard for face models)
        normalized = (resized.astype(np.float32) - 127.5) / 127.5
        
        processed.append(normalized)
    
    return np.array(processed)


def create_train_val_test_split(
    images: np.ndarray, 
    labels: np.ndarray,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> dict:
    """
    Split dataset into train/val/test sets.
    
    Ensures each person appears in all splits (stratified where possible).
    """
    print(f"\n📊 Splitting dataset (train: {1-val_ratio-test_ratio:.0%}, val: {val_ratio:.0%}, test: {test_ratio:.0%})...")
    
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        images, labels, 
        test_size=test_ratio,
        stratify=labels,
        random_state=42
    )
    
    # Second split: train vs val
    val_size_adjusted = val_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size_adjusted,
        stratify=y_trainval,
        random_state=42
    )
    
    print(f"   Train: {len(X_train)} images")
    print(f"   Val:   {len(X_val)} images")
    print(f"   Test:  {len(X_test)} images")
    
    return {
        "train": {"images": X_train, "labels": y_train},
        "val": {"images": X_val, "labels": y_val},
        "test": {"images": X_test, "labels": y_test}
    }


def save_processed_data(data: dict, names: np.ndarray, output_dir: Path):
    """Save processed data to disk."""
    print(f"\n💾 Saving processed data to {output_dir}...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save numpy arrays
    for split_name, split_data in data.items():
        np.save(output_dir / f"{split_name}_images.npy", split_data["images"])
        np.save(output_dir / f"{split_name}_labels.npy", split_data["labels"])
    
    # Save label names
    np.save(output_dir / "label_names.npy", names)
    
    # Save metadata
    metadata = {
        "train_size": len(data["train"]["images"]),
        "val_size": len(data["val"]["images"]),
        "test_size": len(data["test"]["images"]),
        "num_classes": len(names),
        "image_size": data["train"]["images"][0].shape,
    }
    
    with open(output_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    print("✅ Data saved successfully!")
    print(f"   Location: {output_dir}")
    
    return metadata


def print_class_distribution(labels: np.ndarray, names: np.ndarray, top_n: int = 10):
    """Print distribution of images per person."""
    unique, counts = np.unique(labels, return_counts=True)
    sorted_idx = np.argsort(counts)[::-1]
    
    print(f"\n📊 Top {top_n} people by image count:")
    print("-" * 40)
    for i in sorted_idx[:top_n]:
        print(f"   {names[unique[i]]}: {counts[i]} images")
    print(f"   ... and {len(unique) - top_n} more")


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess LFW dataset")
    parser.add_argument("--min-faces", type=int, default=20, 
                        help="Minimum faces per person (default: 20)")
    parser.add_argument("--target-size", type=int, default=160,
                        help="Output image size (default: 160)")
    parser.add_argument("--skip-pairs", action="store_true",
                        help="Skip downloading verification pairs")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 50)
    print("🎯 LFW Dataset Download & Preprocessing")
    print("=" * 50)
    
    # Create directories
    LFW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download LFW people
    data = download_lfw_people(min_faces=args.min_faces)
    
    # Print class distribution
    print_class_distribution(data["labels"], data["names"])
    
    # Preprocess images
    processed_images = preprocess_images(data["images"], target_size=args.target_size)
    
    # Create splits
    splits = create_train_val_test_split(processed_images, data["labels"])
    
    # Save processed data
    metadata = save_processed_data(splits, data["names"], PROCESSED_DIR)
    
    # Download verification pairs (optional)
    if not args.skip_pairs:
        pairs_data = download_lfw_pairs()
        np.save(PROCESSED_DIR / "test_pairs.npy", pairs_data["pairs"])
        np.save(PROCESSED_DIR / "test_pairs_labels.npy", pairs_data["labels"])
        print(f"✅ Saved {len(pairs_data['pairs'])} verification pairs")
    
    print("\n" + "=" * 50)
    print("✅ Dataset preparation complete!")
    print("=" * 50)
    print(f"\n📁 Processed data saved to: {PROCESSED_DIR}")
    print(f"   • {metadata['train_size']} training images")
    print(f"   • {metadata['val_size']} validation images")
    print(f"   • {metadata['test_size']} test images")
    print(f"   • {metadata['num_classes']} unique people")
    print(f"\n👉 NEXT STEP: Run 'python scripts/train_model.py' to train the model")


if __name__ == "__main__":
    main()
