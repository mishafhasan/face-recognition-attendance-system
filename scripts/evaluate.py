"""
Evaluate MobileFaceNet on LFW and Sample Images

Evaluation pipeline:
  - Load trained MobileFaceNet backbone (best_backbone.pth or checkpoint)
  - LFW pair verification (accuracy, AUC, TAR@FAR=0.001)
  - Sample image comparison
  - ROC curve and similarity distribution visualization
  - Inference speed benchmarking

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model-path models/checkpoints/best_backbone.pth
    python scripts/evaluate.py --benchmark-only
    python scripts/evaluate.py --lfw-dir data/lfw/lfw-deepfunneled

Requirements:
    pip install torch albumentations scikit-learn matplotlib
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Environment Detection
# ---------------------------------------------------------------------------
try:
    from google.colab import drive  # noqa: F401

    IN_COLAB = True
    PROJECT_ROOT = Path("/content/face_recognition")
except ImportError:
    IN_COLAB = False
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))
from scripts.model import MobileFaceNet  # noqa: E402


# =============================================================================
# Preprocessing
# =============================================================================

def get_eval_transform():
    """Return evaluation preprocessing pipeline."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose([
        A.Resize(112, 112),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])


def preprocess_image(img_path, transform=None):
    """Load and preprocess a single image."""
    if transform is None:
        transform = get_eval_transform()
    img = np.array(Image.open(img_path).convert("RGB"))
    img = transform(image=img)["image"]
    return img.unsqueeze(0)  # Add batch dimension


@torch.no_grad()
def get_embedding(model, img_path, device, transform=None):
    """Extract 512-D embedding from a single image."""
    img = preprocess_image(img_path, transform).to(device)
    emb = model(img)
    return emb.cpu().numpy().flatten()


@torch.no_grad()
def get_embeddings_batch(model, img_paths, device, batch_size=32, transform=None):
    """Extract embeddings from a batch of images."""
    if transform is None:
        transform = get_eval_transform()

    embeddings = []
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i : i + batch_size]
        batch_imgs = torch.cat([preprocess_image(p, transform) for p in batch_paths]).to(device)
        batch_embs = model(batch_imgs)
        embeddings.append(batch_embs.cpu().numpy())

    return np.vstack(embeddings)


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings."""
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))


def evaluate_pairs(embeddings1, embeddings2, labels):
    """
    Evaluate face verification on pairs.

    Args:
        embeddings1: (N, 512) array of first embeddings
        embeddings2: (N, 512) array of second embeddings
        labels: (N,) array of labels (1=same, 0=different)

    Returns:
        dict with accuracy, AUC, threshold, TAR@FAR, FPR, TPR
    """
    from sklearn.metrics import roc_curve, auc

    # Compute similarities
    similarities = np.array([
        compute_cosine_similarity(e1, e2)
        for e1, e2 in zip(embeddings1, embeddings2)
    ])

    # ROC curve
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    roc_auc = auc(fpr, tpr)

    # Find optimal threshold (maximize accuracy)
    best_acc = 0
    best_threshold = 0.5
    for t in np.arange(-0.2, 1.01, 0.01):
        preds = (similarities >= t).astype(int)
        acc = np.mean(preds == labels) * 100
        if acc > best_acc:
            best_acc = acc
            best_threshold = t

    # TAR @ FAR = 0.001
    far_target = 0.001
    idx = np.argmin(np.abs(fpr - far_target))
    tar_at_far0001 = tpr[idx] * 100

    return {
        "accuracy": best_acc,
        "auc": roc_auc,
        "threshold": best_threshold,
        "tar@far0.001": tar_at_far0001,
        "fpr": fpr,
        "tpr": tpr,
    }


# =============================================================================
# LFW Evaluation
# =============================================================================

def load_lfw_pairs(pairs_file, lfw_dir):
    """
    Load LFW pairs from pairs.txt file.

    Format:
    - Same person: name n1 n2
    - Different: name1 n1 name2 n2
    """
    pairs = []
    labels = []

    pairs_file = Path(pairs_file)
    lfw_dir = Path(lfw_dir)

    if not pairs_file.exists():
        print(f"  Pairs file not found: {pairs_file}")
        return [], []

    with open(pairs_file) as f:
        lines = f.readlines()[1:]  # Skip header

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            name, n1, n2 = parts
            img1 = lfw_dir / name / f"{name}_{int(n1):04d}.jpg"
            img2 = lfw_dir / name / f"{name}_{int(n2):04d}.jpg"
            label = 1
        elif len(parts) == 4:
            name1, n1, name2, n2 = parts
            img1 = lfw_dir / name1 / f"{name1}_{int(n1):04d}.jpg"
            img2 = lfw_dir / name2 / f"{name2}_{int(n2):04d}.jpg"
            label = 0
        else:
            continue

        if img1.exists() and img2.exists():
            pairs.append((img1, img2))
            labels.append(label)

    return pairs, labels


def evaluate_lfw(model, device, lfw_dir, pairs_file, transform=None):
    """Run full LFW evaluation."""
    from tqdm.auto import tqdm

    pairs, labels = load_lfw_pairs(pairs_file, lfw_dir)
    if not pairs:
        print("  No LFW pairs found. Skipping.")
        return None

    print(f"  LFW pairs: {len(pairs)} (pos={sum(labels)}, neg={len(labels) - sum(labels)})")

    # Extract embeddings
    emb1_list, emb2_list = [], []
    for img1, img2 in tqdm(pairs, desc="Extracting LFW embeddings"):
        e1 = get_embedding(model, img1, device, transform)
        e2 = get_embedding(model, img2, device, transform)
        emb1_list.append(e1)
        emb2_list.append(e2)

    embeddings1 = np.array(emb1_list)
    embeddings2 = np.array(emb2_list)
    labels_array = np.array(labels)

    results = evaluate_pairs(embeddings1, embeddings2, labels_array)

    print(f"\n  LFW Results:")
    print(f"  {'=' * 40}")
    print(f"    Accuracy:      {results['accuracy']:.2f}%")
    print(f"    AUC:           {results['auc']:.4f}")
    print(f"    Best Threshold: {results['threshold']:.3f}")
    print(f"    TAR@FAR=0.001: {results['tar@far0.001']:.2f}%")

    return results, embeddings1, embeddings2, labels


# =============================================================================
# Sample Testing
# =============================================================================

def test_on_samples(model, device, test_dir, transform=None):
    """Test model on sample images in a directory."""
    test_dir = Path(test_dir)
    if not test_dir.exists():
        print(f"  Test directory not found: {test_dir}")
        return

    images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    if len(images) < 2:
        print("  Need at least 2 images for comparison")
        return

    print(f"\n  Testing on {len(images)} images from {test_dir}")

    embeddings = {}
    for img_path in images:
        emb = get_embedding(model, img_path, device, transform)
        embeddings[img_path.name] = emb
        print(f"    {img_path.name}: embedding {emb.shape}")

    print(f"\n  Similarity Matrix:")
    names = list(embeddings.keys())
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i < j:
                sim = compute_cosine_similarity(embeddings[n1], embeddings[n2])
                match = "MATCH" if sim > 0.5 else "NO MATCH"
                print(f"    {n1} vs {n2}: {sim:.4f} {match}")


# =============================================================================
# Visualization
# =============================================================================

def plot_roc_curve(results, save_path=None):
    """Plot ROC curve from evaluation results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    plt.plot(results["fpr"], results["tpr"], label=f"LFW (AUC={results['auc']:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Face Verification")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.annotate(f"Accuracy: {results['accuracy']:.2f}%", xy=(0.6, 0.2), fontsize=12)
    plt.annotate(f"Threshold: {results['threshold']:.3f}", xy=(0.6, 0.15), fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  ROC curve saved to {save_path}")
    plt.close()


def plot_similarity_distribution(embeddings1, embeddings2, labels, threshold=None, save_path=None):
    """Plot similarity distribution for same/different pairs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    same_sims, diff_sims = [], []
    for e1, e2, label in zip(embeddings1, embeddings2, labels):
        sim = compute_cosine_similarity(e1, e2)
        if label == 1:
            same_sims.append(sim)
        else:
            diff_sims.append(sim)

    plt.figure(figsize=(10, 5))
    plt.hist(same_sims, bins=50, alpha=0.7, label="Same Person", color="green")
    plt.hist(diff_sims, bins=50, alpha=0.7, label="Different Person", color="red")
    if threshold is not None:
        plt.axvline(threshold, color="blue", linestyle="--", label=f"Threshold ({threshold:.2f})")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.title("Similarity Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Distribution saved to {save_path}")
    plt.close()


# =============================================================================
# Inference Benchmark
# =============================================================================

def benchmark_inference(model, device, num_iterations=100, batch_sizes=(1, 8, 32)):
    """Benchmark inference speed at various batch sizes."""
    model.eval()

    print(f"\n  Inference Speed Benchmark:")
    print(f"  {'=' * 45}")

    for bs in batch_sizes:
        dummy = torch.randn(bs, 3, 112, 112).to(device)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        total_time = time.time() - start
        avg_ms = total_time / num_iterations * 1000
        throughput = bs / (avg_ms / 1000)
        print(f"    Batch {bs:>2}: {avg_ms:.2f} ms ({throughput:.1f} img/s)")

    print(f"  {'=' * 45}")


# =============================================================================
# Model Loading
# =============================================================================

def load_model(model_path, device, emb_dim=512):
    """Load MobileFaceNet backbone from checkpoint or state dict."""
    model = MobileFaceNet(emb_dim=emb_dim).to(device)

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"  Model not found: {model_path}")
        print(f"  Using random weights (for testing only)")
        model.eval()
        return model

    state = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(state, dict) and "model" in state:
        # Full checkpoint - extract backbone
        backbone_state = {
            k.replace("backbone.", ""): v
            for k, v in state["model"].items()
            if k.startswith("backbone.")
        }
        if backbone_state:
            model.load_state_dict(backbone_state)
            print(f"  Loaded backbone from checkpoint: {model_path.name}")
        else:
            model.load_state_dict(state["model"])
            print(f"  Loaded full model from checkpoint: {model_path.name}")
    else:
        # Direct state dict
        model.load_state_dict(state)
        print(f"  Loaded model: {model_path.name}")

    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate MobileFaceNet")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model weights (default: models/checkpoints/best_backbone.pth)")
    parser.add_argument("--lfw-dir", type=str, default=None,
                        help="LFW images directory (default: data/lfw/lfw-deepfunneled)")
    parser.add_argument("--lfw-pairs", type=str, default=None,
                        help="LFW pairs.txt file (default: data/lfw/pairs.txt)")
    parser.add_argument("--test-dir", type=str, default=None,
                        help="Directory with test images for pairwise comparison")
    parser.add_argument("--benchmark-only", action="store_true",
                        help="Only run inference speed benchmark")
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--save-plots", action="store_true", help="Save ROC and distribution plots")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Default paths
    checkpoint_dir = PROJECT_ROOT / "models" / "checkpoints"
    data_dir = PROJECT_ROOT / "data"

    model_path = Path(args.model_path) if args.model_path else checkpoint_dir / "best_backbone.pth"
    lfw_dir = Path(args.lfw_dir) if args.lfw_dir else data_dir / "lfw" / "lfw-deepfunneled"
    lfw_pairs = Path(args.lfw_pairs) if args.lfw_pairs else data_dir / "lfw" / "pairs.txt"
    test_dir = Path(args.test_dir) if args.test_dir else data_dir / "test_images"

    # Load model
    model = load_model(model_path, device, args.embedding_dim)
    transform = get_eval_transform()

    # Benchmark
    benchmark_inference(model, device)

    if args.benchmark_only:
        return

    # LFW evaluation
    if lfw_dir.exists() and lfw_pairs.exists():
        result = evaluate_lfw(model, device, lfw_dir, lfw_pairs, transform)
        if result is not None:
            results, emb1, emb2, labels = result
            if args.save_plots:
                plot_roc_curve(results, save_path=checkpoint_dir / "roc_curve.png")
                plot_similarity_distribution(
                    emb1, emb2, labels,
                    threshold=results["threshold"],
                    save_path=checkpoint_dir / "similarity_distribution.png",
                )
    else:
        print(f"\n  LFW not found at {lfw_dir}")
        print(f"  Download: wget http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz")

    # Sample testing
    if test_dir.exists():
        test_on_samples(model, device, test_dir, transform)

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Model: MobileFaceNet")
    print(f"Parameters: ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"Embedding: {args.embedding_dim}-D, L2-normalized")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
