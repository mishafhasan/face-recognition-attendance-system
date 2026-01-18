"""
Model Evaluation Script

Evaluates the trained face embedding model on LFW verification pairs.

Usage:
    python evaluate_model.py
    python evaluate_model.py --model models/best_model.pth
    python evaluate_model.py --find-threshold
"""
import argparse
import sys
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.ml.model import FaceEmbeddingModel, create_model
from backend.app.ml.dataset import load_processed_data, PairsFaceDataset
from torch.utils.data import DataLoader


# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "lfw" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def load_model(model_path: Path, device: str) -> FaceEmbeddingModel:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    embedding_dim = checkpoint.get("embedding_dim", 128)
    model = create_model(embedding_dim=embedding_dim, pretrained=False, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model


def compute_embeddings_for_pairs(
    model: FaceEmbeddingModel,
    pairs: np.ndarray,
    device: str,
    batch_size: int = 32
) -> tuple:
    """
    Compute embeddings for all pairs in the dataset.
    
    Returns:
        Tuple of (embeddings1, embeddings2) arrays
    """
    model.eval()
    
    embeddings1 = []
    embeddings2 = []
    
    num_pairs = len(pairs)
    
    with torch.no_grad():
        for i in tqdm(range(0, num_pairs, batch_size), desc="Computing embeddings"):
            batch_pairs = pairs[i:i + batch_size]
            
            # Process first images
            imgs1 = []
            imgs2 = []
            
            for pair in batch_pairs:
                img1, img2 = pair[0], pair[1]
                
                # Preprocess
                if img1.max() > 1.0:
                    img1 = (img1.astype(np.float32) - 127.5) / 127.5
                    img2 = (img2.astype(np.float32) - 127.5) / 127.5
                
                # Resize if needed
                import cv2
                if img1.shape[0] != 160:
                    img1 = cv2.resize(img1, (160, 160))
                    img2 = cv2.resize(img2, (160, 160))
                
                imgs1.append(img1.transpose(2, 0, 1))
                imgs2.append(img2.transpose(2, 0, 1))
            
            # To tensors
            batch1 = torch.from_numpy(np.array(imgs1)).float().to(device)
            batch2 = torch.from_numpy(np.array(imgs2)).float().to(device)
            
            # Get embeddings
            emb1 = model(batch1).cpu().numpy()
            emb2 = model(batch2).cpu().numpy()
            
            embeddings1.append(emb1)
            embeddings2.append(emb2)
    
    return np.vstack(embeddings1), np.vstack(embeddings2)


def compute_distances(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """Compute L2 distances between embedding pairs."""
    return np.sqrt(np.sum((embeddings1 - embeddings2) ** 2, axis=1))


def compute_cosine_similarity(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between embedding pairs."""
    # Already L2 normalized, so dot product = cosine similarity
    return np.sum(embeddings1 * embeddings2, axis=1)


def find_best_threshold(distances: np.ndarray, labels: np.ndarray) -> tuple:
    """
    Find the best threshold for verification.
    
    Returns:
        Tuple of (best_threshold, best_accuracy)
    """
    thresholds = np.arange(0, 2, 0.01)
    best_acc = 0
    best_thresh = 0
    
    for thresh in thresholds:
        predictions = (distances < thresh).astype(int)
        acc = accuracy_score(labels, predictions)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    
    return best_thresh, best_acc


def plot_roc_curve(distances: np.ndarray, labels: np.ndarray, save_path: Path):
    """Plot and save ROC curve."""
    # For ROC: lower distance = more likely same person
    # So we negate distances to get "scores"
    scores = -distances
    
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'r--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - LFW Verification')
    ax.legend(loc='lower right')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return roc_auc


def plot_distance_distribution(
    distances: np.ndarray, 
    labels: np.ndarray, 
    threshold: float,
    save_path: Path
):
    """Plot distribution of distances for same vs different pairs."""
    same_distances = distances[labels == 1]
    diff_distances = distances[labels == 0]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.hist(same_distances, bins=50, alpha=0.7, label='Same Person', color='green')
    ax.hist(diff_distances, bins=50, alpha=0.7, label='Different Person', color='red')
    ax.axvline(x=threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.3f}')
    
    ax.set_xlabel('L2 Distance')
    ax.set_ylabel('Count')
    ax.set_title('Distance Distribution')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate face embedding model")
    parser.add_argument("--model", type=str, default=str(MODELS_DIR / "best_model.pth"),
                        help="Path to model checkpoint")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")
    
    args = parser.parse_args()
    
    # Setup device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"\n🖥️  Using device: {device}")
    
    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Run 'python scripts/train_model.py' first!")
        return
    
    # Check data exists
    if not DATA_DIR.exists():
        print(f"❌ Data not found: {DATA_DIR}")
        print("   Run 'python scripts/download_lfw.py' first!")
        return
    
    print("\n" + "=" * 50)
    print("📊 Face Embedding Model Evaluation")
    print("=" * 50)
    
    # Load model
    print(f"\n📥 Loading model from {model_path}...")
    model = load_model(model_path, device)
    print("   ✅ Model loaded")
    
    # Load test pairs
    print("\n📂 Loading LFW verification pairs...")
    pairs_path = DATA_DIR / "test_pairs.npy"
    labels_path = DATA_DIR / "test_pairs_labels.npy"
    
    if not pairs_path.exists():
        print("❌ Test pairs not found. Run download_lfw.py again without --skip-pairs")
        return
    
    pairs = np.load(pairs_path)
    labels = np.load(labels_path)
    print(f"   Loaded {len(pairs)} pairs ({sum(labels)} same, {len(labels) - sum(labels)} different)")
    
    # Compute embeddings
    print("\n🔄 Computing embeddings...")
    emb1, emb2 = compute_embeddings_for_pairs(model, pairs, device, args.batch_size)
    
    # Compute distances
    print("\n📐 Computing distances...")
    distances = compute_distances(emb1, emb2)
    
    # Find best threshold
    print("\n🔍 Finding best threshold...")
    best_threshold, best_accuracy = find_best_threshold(distances, labels)
    
    # Compute metrics
    print("\n📈 Computing metrics...")
    predictions = (distances < best_threshold).astype(int)
    accuracy = accuracy_score(labels, predictions)
    
    # Plot ROC curve
    roc_auc = plot_roc_curve(distances, labels, MODELS_DIR / "roc_curve.png")
    
    # Plot distance distribution
    plot_distance_distribution(distances, labels, best_threshold, MODELS_DIR / "distance_distribution.png")
    
    # Print results
    print("\n" + "=" * 50)
    print("📊 EVALUATION RESULTS")
    print("=" * 50)
    print(f"\n   📍 Best Threshold: {best_threshold:.4f}")
    print(f"   ✅ Accuracy: {accuracy:.2%}")
    print(f"   📈 AUC: {roc_auc:.4f}")
    print(f"\n   Distance Statistics:")
    print(f"      Same Person:      {distances[labels==1].mean():.4f} ± {distances[labels==1].std():.4f}")
    print(f"      Different Person: {distances[labels==0].mean():.4f} ± {distances[labels==0].std():.4f}")
    
    print(f"\n📁 Plots saved to: {MODELS_DIR}")
    print(f"   • roc_curve.png")
    print(f"   • distance_distribution.png")
    
    # Save evaluation results
    import json
    results = {
        "model_path": str(model_path),
        "num_pairs": len(pairs),
        "best_threshold": float(best_threshold),
        "accuracy": float(accuracy),
        "auc": float(roc_auc),
        "same_distance_mean": float(distances[labels==1].mean()),
        "same_distance_std": float(distances[labels==1].std()),
        "diff_distance_mean": float(distances[labels==0].mean()),
        "diff_distance_std": float(distances[labels==0].std())
    }
    
    with open(MODELS_DIR / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n👉 NEXT STEP: Build the Next.js demo app with 'npx create-next-app frontend'")


if __name__ == "__main__":
    main()
