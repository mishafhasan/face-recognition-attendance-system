"""
PyTorch Dataset Classes for Face Recognition Training

Provides TripletFaceDataset for training with triplet loss and
PairsFaceDataset for evaluation on LFW verification pairs.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import random
from typing import Tuple, Optional
import cv2


class TripletFaceDataset(Dataset):
    """
    Dataset that generates triplets (anchor, positive, negative) for training.
    
    For each anchor image, selects:
    - Positive: Another image of the same person
    - Negative: Image of a different person
    """
    
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform=None,
        triplets_per_image: int = 1
    ):
        """
        Args:
            images: Array of preprocessed face images [N, H, W, C]
            labels: Array of person labels [N]
            transform: Optional torchvision transforms
            triplets_per_image: Number of triplets to generate per anchor
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        self.triplets_per_image = triplets_per_image
        
        # Build index: label -> list of image indices
        self.label_to_indices = {}
        for idx, label in enumerate(labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
        
        # Only keep labels with at least 2 images (for positive pairs)
        self.valid_labels = [
            label for label, indices in self.label_to_indices.items()
            if len(indices) >= 2
        ]
        
        # Get all labels for negative sampling
        self.all_labels = list(self.label_to_indices.keys())
        
    def __len__(self):
        return len(self.images) * self.triplets_per_image
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get anchor index
        anchor_idx = idx % len(self.images)
        anchor_label = self.labels[anchor_idx]
        
        # Only generate valid triplets if this label has enough images
        if anchor_label not in self.valid_labels:
            # Fall back to a random valid anchor
            anchor_label = random.choice(self.valid_labels)
            anchor_idx = random.choice(self.label_to_indices[anchor_label])
        
        # Get positive (same person, different image)
        positive_indices = [
            i for i in self.label_to_indices[anchor_label] 
            if i != anchor_idx
        ]
        positive_idx = random.choice(positive_indices)
        
        # Get negative (different person)
        negative_label = random.choice([
            l for l in self.all_labels if l != anchor_label
        ])
        negative_idx = random.choice(self.label_to_indices[negative_label])
        
        # Load images
        anchor = self._load_image(anchor_idx)
        positive = self._load_image(positive_idx)
        negative = self._load_image(negative_idx)
        
        return anchor, positive, negative
    
    def _load_image(self, idx: int) -> torch.Tensor:
        """Load and transform a single image."""
        img = self.images[idx]
        
        # Convert from [H, W, C] to [C, H, W]
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        
        if self.transform:
            img = self.transform(img)
        
        return img


class PairsFaceDataset(Dataset):
    """
    Dataset for LFW verification pairs evaluation.
    
    Each item is a pair of images and a label (1 = same person, 0 = different).
    """
    
    def __init__(
        self,
        pairs: np.ndarray,
        labels: np.ndarray,
        transform=None
    ):
        """
        Args:
            pairs: Array of image pairs [N, 2, H, W, C]
            labels: Array of labels [N] (1 = same, 0 = different)
            transform: Optional torchvision transforms
        """
        self.pairs = pairs
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, int]:
        pair = self.pairs[idx]
        label = self.labels[idx]
        
        # Process both images in the pair
        img1 = self._process_image(pair[0])
        img2 = self._process_image(pair[1])
        
        return img1, img2, label
    
    def _process_image(self, img: np.ndarray) -> torch.Tensor:
        """Process a single image."""
        # Normalize if needed
        if img.max() > 1.0:
            img = (img.astype(np.float32) - 127.5) / 127.5
        
        # Resize if needed (LFW pairs are 250x250)
        if img.shape[0] != 160:
            img = cv2.resize(img, (160, 160))
        
        # Convert to tensor [C, H, W]
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        
        if self.transform:
            img = self.transform(img)
        
        return img


def load_processed_data(data_dir: Path) -> dict:
    """
    Load preprocessed LFW data from numpy files.
    
    Args:
        data_dir: Directory containing processed .npy files
        
    Returns:
        Dictionary with train/val/test data
    """
    data = {}
    
    for split in ["train", "val", "test"]:
        images = np.load(data_dir / f"{split}_images.npy")
        labels = np.load(data_dir / f"{split}_labels.npy")
        data[split] = {"images": images, "labels": labels}
    
    # Load label names
    data["names"] = np.load(data_dir / "label_names.npy")
    
    # Load pairs if available
    pairs_path = data_dir / "test_pairs.npy"
    if pairs_path.exists():
        data["pairs"] = np.load(pairs_path)
        data["pairs_labels"] = np.load(data_dir / "test_pairs_labels.npy")
    
    return data


def create_data_loaders(
    data: dict,
    batch_size: int = 32,
    num_workers: int = 0
) -> dict:
    """
    Create PyTorch DataLoaders for training and evaluation.
    
    Args:
        data: Dictionary with train/val/test data
        batch_size: Batch size for training
        num_workers: Number of worker processes
        
    Returns:
        Dictionary with DataLoaders
    """
    from torch.utils.data import DataLoader
    
    loaders = {}
    
    # Training loader (triplets)
    train_dataset = TripletFaceDataset(
        data["train"]["images"],
        data["train"]["labels"],
        triplets_per_image=2  # Generate more triplets per epoch
    )
    loaders["train"] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Validation loader (triplets)
    val_dataset = TripletFaceDataset(
        data["val"]["images"],
        data["val"]["labels"]
    )
    loaders["val"] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    # Test pairs loader (if available)
    if "pairs" in data:
        pairs_dataset = PairsFaceDataset(
            data["pairs"],
            data["pairs_labels"]
        )
        loaders["pairs"] = DataLoader(
            pairs_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    
    return loaders


if __name__ == "__main__":
    # Test the dataset classes
    from pathlib import Path
    
    data_dir = Path(__file__).parent.parent / "data" / "lfw" / "processed"
    
    if data_dir.exists():
        print("Loading processed data...")
        data = load_processed_data(data_dir)
        print(f"Train: {len(data['train']['images'])} images")
        print(f"Val: {len(data['val']['images'])} images")
        print(f"Test: {len(data['test']['images'])} images")
        
        print("\nCreating data loaders...")
        loaders = create_data_loaders(data, batch_size=16)
        
        # Test a batch
        anchor, positive, negative = next(iter(loaders["train"]))
        print(f"Triplet shapes: {anchor.shape}, {positive.shape}, {negative.shape}")
    else:
        print(f"Data directory not found: {data_dir}")
        print("Run download_lfw.py first!")
