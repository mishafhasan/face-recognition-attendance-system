"""
Train MobileFaceNet + ArcFace on VGGFace2-112x112

Full training pipeline with:
  - Mixed precision (AMP) - auto-disabled on CPU
  - ArcFace margin warmup (0 → 0.5 over 5 epochs) to avoid cold-start collapse
  - Warmup (fixed steps) + Multi-Step LR decay (industry-standard for face recognition)
  - Auto-checkpoint every 500 steps (Colab disconnect protection)
  - Class-compatible auto-split for correct ArcFace validation
  - Embedding-based pair verification (industry standard metric)
  - Raw cosine accuracy tracking (true model quality measure)

BUG FIXES vs old version:
  - FIXED: --resume flag was parsed but NEVER used; fresh runs now start from epoch 1
  - FIXED: use_amp hardcoded True even on CPU (GradScaler would crash without CUDA)
  - FIXED: base_lr=0.01 too low; now 0.1 (insightface default for single-GPU)
  - FIXED: warmup=1 full epoch kept LR near-zero all of epoch 1; now fixed 1000 steps
  - FIXED: cosine decay only; now uses multi-step LR (better convergence for face recognition)
  - FIXED: autocast/GradScaler hardcoded to "cuda"; now dynamic based on actual device
  - FIXED: accuracy computed on margin-penalized logits (always 0%); uses raw cosine

Usage:
    python scripts/train.py                          # Fresh start (ignores old checkpoints)
    python scripts/train.py --resume                 # Resume from latest checkpoint
    python scripts/train.py --epochs 30 --lr 0.1
    python scripts/train.py --lr-schedule cosine     # Use cosine LR (default: multistep)

Requirements:
    pip install torch albumentations tqdm scikit-learn
"""

import argparse
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset, Subset

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

# Import model architecture from shared module
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.model import ArcFace, MobileArcFace, MobileFaceNet  # noqa: E402


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration with GPU optimizations.

    Key changes vs previous version:
      - use_amp: auto-detected (False on CPU to prevent GradScaler crash)
      - base_lr: 0.1 (was 0.01 — 10x too low, causing near-zero gradient updates)
      - warmup_steps: 1000 fixed steps (was 1 full epoch ~12K steps — LR was 0 all epoch 1)
      - lr_schedule: multistep (was cosine — multistep gives flat LR phases for face rec)
      - auto_resume: False (was always-True — caused silent 0-epoch runs after checkpoint)
    """

    # Paths
    train_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "train")
    val_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "val")
    checkpoint_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models" / "checkpoints")

    # Model
    embedding_dim: int = 512

    # ArcFace
    arcface_scale: float = 64.0
    arcface_margin: float = 0.5

    # Margin warmup: prevents training collapse at start (ArcFace with full margin = very high loss at init)
    margin_warmup_epochs: int = 5
    arcface_margin_start: float = 0.0
    arcface_margin_end: float = 0.5

    # Training
    batch_size: int = 256
    num_epochs: int = 25

    # FIXED: was 0.01 — 10x too low for SGD ArcFace training
    # Standard (insightface, official MobileFaceNet): 0.1 for single-GPU
    # With multi-step decay: 0.1 -> 0.01 -> 0.001 -> 0.0001
    base_lr: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9

    # LR Schedule: "multistep" (recommended) or "cosine"
    # Multi-step: standard for all insightface/ArcFace models; gives stable flat-LR phases
    # Cosine: smooth but can decay LR too aggressively before convergence in early epochs
    lr_schedule: str = "multistep"

    # Multi-step LR milestones (epoch numbers) and decay factor
    # Standard for VGGFace2/MS1M: drop at 10, 18, 22 out of 25 epochs
    lr_milestones: list = field(default_factory=lambda: [10, 18, 22])
    lr_gamma: float = 0.1          # Each milestone: LR × 0.1

    # Cosine schedule parameters (only used if lr_schedule="cosine")
    min_lr: float = 1e-5           # Raised from 1e-6 to avoid near-zero at end

    # FIXED: warmup is now fixed steps, not epoch-based.
    # Previous: 1 full epoch warmup (~12K steps) meant LR ≈ 0 through ALL of epoch 1.
    # Fixed: 1000 steps warmup = a few minutes, then LR reaches base_lr.
    warmup_steps: int = 1000

    # Mixed Precision
    # FIXED: was hardcoded True — crashed with GradScaler("cuda") when no CUDA available
    use_amp: bool = field(default_factory=lambda: torch.cuda.is_available())

    # Gradient clipping
    grad_clip: float = 5.0

    # Checkpointing (Colab/disconnect protection)
    checkpoint_every_n_steps: int = 500
    keep_last_n_checkpoints: int = 3

    # FIXED: was always-True (always resumed if checkpoint existed).
    # Previous bug: checkpoint from epoch 25 exists → `range(25, 25)` = 0 iterations → no training
    # Fixed: resume only when explicitly requested via --resume flag
    auto_resume: bool = False

    # Data loading - GPU optimization settings
    num_workers: int = 6 if IN_COLAB else 8
    pin_memory: bool = True
    prefetch_factor: int = 3
    persistent_workers: bool = True
    non_blocking: bool = True


# =============================================================================
# Dataset
# =============================================================================

class FaceDataset(Dataset):
    """Face dataset with identity labels and albumentations transforms."""

    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        if self.root_dir.exists():
            for idx, class_dir in enumerate(sorted(self.root_dir.iterdir())):
                if class_dir.is_dir():
                    self.class_to_idx[class_dir.name] = idx
                    for img_path in class_dir.glob("*.jpg"):
                        self.samples.append((img_path, idx))
                    for img_path in class_dir.glob("*.png"):
                        self.samples.append((img_path, idx))

        self.num_classes = len(self.class_to_idx)
        print(f"  Dataset: {len(self.samples):,} images, {self.num_classes:,} classes from {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label


def get_transforms():
    """Return train and val augmentation pipelines (albumentations v2.x)."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.Rotate(limit=10, border_mode=0, p=0.3),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])

    return train_transform, val_transform


# =============================================================================
# Data Loading with Class-Compatibility Check
# =============================================================================

def create_datasets(config: TrainingConfig):
    """
    Create training and validation datasets with class-compatibility handling.

    ArcFace classification-based validation REQUIRES matching class indices.
    If the val set has different identities than train, we auto-split from
    training data (90/10 stratified) for accurate validation metrics.
    """
    from sklearn.model_selection import train_test_split

    train_transform, val_transform = get_transforms()

    train_dataset_full = FaceDataset(config.train_dir, train_transform)
    val_dataset_raw = FaceDataset(config.val_dir, val_transform)

    need_auto_split = False

    if len(val_dataset_raw) == 0:
        need_auto_split = True
        print("  No validation data found -> will auto-split from training data")
    elif len(train_dataset_full) > 0:
        train_classes = set(train_dataset_full.class_to_idx.keys())
        val_classes = set(val_dataset_raw.class_to_idx.keys())
        if val_classes != train_classes:
            need_auto_split = True
            overlap = train_classes & val_classes
            print(f"  Validation identities DIFFER from training!")
            print(f"    Train: {len(train_classes):,}, Val: {len(val_classes):,}, Overlap: {len(overlap):,}")
            print(f"    -> Auto-splitting from training data for accurate ArcFace metrics")

    if need_auto_split and len(train_dataset_full) > 0:
        print("  Creating stratified train/val split (90/10) from training data...")

        all_labels = [label for _, label in train_dataset_full.samples]
        label_counts = Counter(all_labels)
        single_sample_classes = {label for label, count in label_counts.items() if count < 2}

        if single_sample_classes:
            print(f"    {len(single_sample_classes):,} classes with 1 sample -> assigned to train")
            single_sample_indices = [i for i, label in enumerate(all_labels) if label in single_sample_classes]
            multi_sample_indices = [i for i, label in enumerate(all_labels) if label not in single_sample_classes]
            multi_sample_labels = [all_labels[i] for i in multi_sample_indices]

            train_multi, val_multi = train_test_split(
                multi_sample_indices, test_size=0.1,
                stratify=multi_sample_labels, random_state=42,
            )
            train_indices = train_multi + single_sample_indices
            val_indices = val_multi
        else:
            train_indices, val_indices = train_test_split(
                range(len(train_dataset_full)), test_size=0.1,
                stratify=all_labels, random_state=42,
            )

        print(f"    Train: {len(train_indices):,}, Val: {len(val_indices):,}")

        train_dataset = Subset(train_dataset_full, train_indices)
        val_dataset_full = FaceDataset(config.train_dir, val_transform)
        val_dataset = Subset(val_dataset_full, val_indices)
        num_classes = train_dataset_full.num_classes
    else:
        train_dataset = train_dataset_full
        val_dataset = val_dataset_raw
        num_classes = train_dataset_full.num_classes if hasattr(train_dataset_full, "num_classes") else train_dataset.num_classes

    return train_dataset, val_dataset, num_classes


def create_dataloaders(train_dataset, val_dataset, config: TrainingConfig):
    """Create optimized DataLoader instances."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
    )

    print(f"  Train batches: {len(train_loader):,}, Val batches: {len(val_loader):,}")
    return train_loader, val_loader


# =============================================================================
# LR Schedule
# =============================================================================

def get_lr_cosine(step, warmup_steps, total_steps, base_lr, min_lr=1e-5):
    """Warmup (fixed steps) + cosine decay learning rate schedule."""
    if step < warmup_steps:
        # Linear warmup from 0 to base_lr over warmup_steps
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def get_lr_multistep(epoch, base_lr, milestones, gamma=0.1):
    """
    Multi-step learning rate schedule (industry standard for ArcFace training).

    LR = base_lr * gamma^(# milestones passed)
    e.g., base_lr=0.1, milestones=[10,18,22], gamma=0.1:
      epochs 0-9:  LR = 0.1
      epochs 10-17: LR = 0.01
      epochs 18-21: LR = 0.001
      epochs 22+:  LR = 0.0001
    """
    lr = base_lr
    for milestone in milestones:
        if epoch >= milestone:
            lr *= gamma
    return lr


def get_lr(step, epoch, config):
    """Get current learning rate based on configured schedule."""
    if config.lr_schedule == "multistep":
        # Warmup phase (fixed steps at start of training)
        if step < config.warmup_steps:
            target_lr = get_lr_multistep(epoch, config.base_lr, config.lr_milestones, config.lr_gamma)
            return target_lr * step / max(1, config.warmup_steps)
        return get_lr_multistep(epoch, config.base_lr, config.lr_milestones, config.lr_gamma)
    else:  # cosine
        total_steps = getattr(config, '_total_steps', 1000)
        return get_lr_cosine(step, config.warmup_steps, total_steps, config.base_lr, config.min_lr)


def set_lr(optimizer, lr):
    """Set learning rate for all parameter groups."""
    for g in optimizer.param_groups:
        g["lr"] = lr


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(path, model, optimizer, scaler, epoch, step, loss, verbose=True):
    """Save training checkpoint to disk."""
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "loss": loss,
        "timestamp": datetime.now().isoformat(),
    }
    torch.save(checkpoint, path)
    if verbose:
        print(f"  Checkpoint saved: {path.name}")


def load_checkpoint(path, model, optimizer=None, scaler=None, device="cpu"):
    """Load training checkpoint."""
    print(f"  Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    print(f"  Resumed from epoch {ckpt['epoch']}, step {ckpt['step']}")
    return ckpt["epoch"], ckpt["step"]


def find_latest_checkpoint(ckpt_dir):
    """Find most recent checkpoint file."""
    latest = ckpt_dir / "checkpoint_latest.pth"
    if latest.exists():
        return latest
    ckpts = list(ckpt_dir.glob("checkpoint_epoch*.pth"))
    if ckpts:
        return max(ckpts, key=lambda p: p.stat().st_mtime)
    return None


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(model, loader, optimizer, scaler, epoch, global_step, config,
                device, amp_device_type, checkpoint_dir):
    """
    Train for one epoch.

    Accuracy is measured on RAW cosine logits (before ArcFace margin),
    which gives a true picture of the model's discriminative ability.

    FIXES:
      - LR schedule now uses epoch-aware multi-step or cosine (no more 1-epoch warmup)
      - warmup is now based on global_step < config.warmup_steps (fixed steps)
      - AMP device type is dynamic (not hardcoded "cuda")
      - accuracy always from _last_cos (raw cosine), not margin-penalized logits
    """
    from tqdm.auto import tqdm

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=config.non_blocking)
        labels = labels.to(device, non_blocking=config.non_blocking)

        # Update LR: warmup uses fixed global steps, then epoch-based schedule
        lr = get_lr(global_step, epoch, config)
        set_lr(optimizer, lr)

        optimizer.zero_grad()

        with autocast(device_type=amp_device_type, enabled=config.use_amp):
            logits = model(images, labels)
            loss = F.cross_entropy(logits, labels)

        scaler.scale(loss).backward()

        if config.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # Accuracy on RAW cosine (before margin penalty) — the true measure
        # NOTE: Using logits.max() would give 0% acc because ArcFace DECREASES
        # the correct-class logit by the margin (making it harder than other classes).
        # _last_cos has the UNPENALIZED cosine similarities — correct indicator.
        with torch.no_grad():
            if hasattr(model, "head") and model.head._last_cos is not None:
                _, predicted = model.head._last_cos.max(1)
            else:
                # Fallback when _last_cos not populated (e.g., first step, no head)
                _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        total_loss += loss.item()
        global_step += 1

        pbar.set_postfix({
            "loss": f"{total_loss / (batch_idx + 1):.4f}",
            "acc": f"{100.0 * correct / total:.1f}%",
            "lr": f"{lr:.2e}",
        })

        # Periodic checkpoint
        if config.checkpoint_every_n_steps > 0 and global_step % config.checkpoint_every_n_steps == 0:
            save_checkpoint(
                checkpoint_dir / f"checkpoint_step_{global_step}.pth",
                model, optimizer, scaler, epoch, global_step,
                total_loss / (batch_idx + 1), verbose=False,
            )

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, global_step


@torch.no_grad()
def validate(model, loader, config, device, amp_device_type):
    """
    Validate using raw cosine logits (no ArcFace margin penalty).

    FIXED: Now directly uses backbone embeddings + ArcFace weight matrix
    without calling model(images, labels) (which triggers full ArcFace forward
    with unnecessary margin computation and 2x GPU ops).
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    # Get normalized ArcFace weights (class prototypes) once
    w = F.normalize(model.head.weight.float(), p=2, dim=1)
    scale = model.head.s

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=amp_device_type, enabled=config.use_amp):
            # Extract embeddings from backbone only (already L2-normalized)
            emb = model.backbone(images)

        # Cosine similarity (no margin, no scaling yet)
        cos = F.linear(emb.float(), w).clamp(-1 + 1e-7, 1 - 1e-7)
        raw_logits = cos * scale
        loss = F.cross_entropy(raw_logits, labels)

        pred = cos.argmax(dim=1)
        total_correct += (pred == labels).sum().item()
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    return total_loss / max(1, total_samples), 100 * total_correct / max(1, total_samples)


@torch.no_grad()
def validate_embeddings(model, loader, config, device, amp_device_type, num_pairs=3000):
    """
    Embedding-based face verification (industry standard metric).

    Extracts backbone embeddings (already L2-normalised by BN+normalise layer),
    generates positive/negative pairs, and computes verification accuracy,
    TAR@FAR=0.01, and AUC.

    FIXED: amp_device_type parameter (was hardcoded "cuda"); removed redundant
    F.normalize since MobileFaceNet backbone already outputs L2-normalised embeddings.
    """
    import random

    model.eval()
    all_embeddings = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with autocast(device_type=amp_device_type, enabled=config.use_amp):
            if hasattr(model, "backbone"):
                emb = model.backbone(images)
            else:
                emb = model(images)
        # Backbone already outputs L2-normalised 512-D embeddings; cast to float32
        all_embeddings.append(emb.float().cpu())
        all_labels.append(labels.cpu())

    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)

    # Build per-class index
    class_indices = defaultdict(list)
    for idx, label in enumerate(all_labels.tolist()):
        class_indices[label].append(idx)

    valid_classes = [c for c, idxs in class_indices.items() if len(idxs) >= 2]
    all_class_list = list(class_indices.keys())

    if len(valid_classes) < 2:
        return {"verification_accuracy": 0, "auc": 0, "best_threshold": 0.5,
                "tar_at_far001": 0, "pos_sim_mean": 0, "pos_sim_std": 0,
                "neg_sim_mean": 0, "neg_sim_std": 0}

    random.seed(42 + len(all_embeddings))
    half_pairs = num_pairs // 2
    pos_sims, neg_sims = [], []

    for _ in range(half_pairs):
        cls = random.choice(valid_classes)
        i, j = random.sample(class_indices[cls], 2)
        sim = F.cosine_similarity(all_embeddings[i].unsqueeze(0), all_embeddings[j].unsqueeze(0)).item()
        pos_sims.append(sim)

    for _ in range(half_pairs):
        cls1, cls2 = random.sample(all_class_list, 2)
        i = random.choice(class_indices[cls1])
        j = random.choice(class_indices[cls2])
        sim = F.cosine_similarity(all_embeddings[i].unsqueeze(0), all_embeddings[j].unsqueeze(0)).item()
        neg_sims.append(sim)

    all_sims = pos_sims + neg_sims
    all_true = [1] * len(pos_sims) + [0] * len(neg_sims)

    best_acc, best_threshold = 0, 0.5
    for threshold in np.arange(-0.2, 1.01, 0.01):
        preds = [1 if s >= threshold else 0 for s in all_sims]
        acc = sum(p == t for p, t in zip(preds, all_true)) / len(all_true)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    neg_sorted = sorted(neg_sims, reverse=True)
    far_idx = max(1, int(0.01 * len(neg_sorted))) - 1
    threshold_far001 = neg_sorted[far_idx] if neg_sorted else 0.5
    tar_at_far001 = sum(1 for s in pos_sims if s >= threshold_far001) / max(1, len(pos_sims))

    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_true, all_sims)
    except Exception:
        auc = best_acc

    return {
        "verification_accuracy": best_acc * 100,
        "auc": auc,
        "best_threshold": best_threshold,
        "tar_at_far001": tar_at_far001 * 100,
        "pos_sim_mean": float(np.mean(pos_sims)),
        "pos_sim_std": float(np.std(pos_sims)),
        "neg_sim_mean": float(np.mean(neg_sims)),
        "neg_sim_std": float(np.std(neg_sims)),
    }


# =============================================================================
# Main Training Loop
# =============================================================================

def train(config: TrainingConfig):
    """Run the full training pipeline.

    FIXES applied:
    - amp_device_type is now derived from actual device (not hardcoded "cuda")
    - GradScaler uses amp_device_type
    - Checkpoint auto-resume gated on config.auto_resume (Bug #1 fixed)
    - train_epoch() called with new signature (device, amp_device_type)
    - validate() / validate_embeddings() receive amp_device_type
    - warmup_epochs removed; uses config.warmup_steps directly
    - config._total_steps set for cosine LR schedule
    - Data health check (warn if dataset is empty)
    - Per-epoch LR logging for multi-step schedule
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_device_type = "cuda" if device.type == "cuda" else "cpu"

    print(f"\n{'=' * 70}")
    print(f"MobileFaceNet + ArcFace Training")
    print(f"{'=' * 70}")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # GPU optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("  TF32 enabled (Ampere+ GPU)")

    # Data
    print(f"\nLoading datasets...")
    train_dataset, val_dataset, num_classes = create_datasets(config)

    # --- Data health check ---
    if len(train_dataset) == 0:
        print("  ERROR: Training dataset is EMPTY. Check config.train_dir path.")
        print(f"  Expected data at: {config.train_dir}")
        return {}
    print(f"  Train: {len(train_dataset):,} images | {num_classes} identities")
    if val_dataset and len(val_dataset) > 0:
        print(f"  Val:   {len(val_dataset):,} images")

    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, config)

    # Model
    print(f"\nCreating model...")
    model = MobileArcFace(
        num_classes=max(num_classes, 2),
        emb_dim=config.embedding_dim,
        s=config.arcface_scale,
        m=config.arcface_margin,
    ).to(device)

    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        print("  Channels-last memory format enabled")

    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    print(f"  Backbone: {backbone_params:,} params (~{backbone_params * 4 / 1024**2:.1f} MB)")
    print(f"  Head: {head_params:,} params")

    # Optimizer & Scaler
    optimizer = SGD(
        model.parameters(),
        lr=config.base_lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    total_steps = len(train_loader) * config.num_epochs
    # Store on config so get_lr() cosine branch can reference it
    config._total_steps = total_steps
    scaler = GradScaler(amp_device_type, enabled=config.use_amp)

    print(f"\n  Optimizer: SGD (lr={config.base_lr}, momentum={config.momentum})")
    print(f"  LR schedule: {config.lr_schedule}", end="")
    if config.lr_schedule == "multistep":
        print(f" | milestones={config.lr_milestones}, gamma={config.lr_gamma}")
    else:
        print(f" | total_steps={total_steps:,}, warmup_steps={config.warmup_steps}")
    print(f"  AMP: {config.use_amp} ({amp_device_type}), Grad clip: {config.grad_clip}")
    print(f"  Margin warmup: {config.arcface_margin_start} -> {config.arcface_margin_end} over {config.margin_warmup_epochs} epochs")
    print(f"  Auto-resume:  {config.auto_resume}")

    # Resume (gated on config.auto_resume — Bug #1 Fix)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    global_step = 0

    if config.auto_resume:
        latest_ckpt = find_latest_checkpoint(config.checkpoint_dir)
        if latest_ckpt:
            start_epoch, global_step = load_checkpoint(latest_ckpt, model, optimizer, scaler, device)
        else:
            print("  No checkpoint found — starting from scratch.")
    else:
        print("  Auto-resume disabled — training from scratch (epoch 0).")

    if start_epoch >= config.num_epochs:
        print(f"\n  WARNING: start_epoch ({start_epoch}) >= num_epochs ({config.num_epochs}).")
        print(f"  Nothing to train. Use --epochs to set a higher epoch count.")
        return {}

    # Training loop
    best_val_acc = 0.0
    best_train_acc = 0.0
    best_embed_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "embed_acc": []}
    has_val = val_loader is not None and val_dataset is not None and len(val_dataset) > 0

    print(f"\n{'=' * 70}")
    print(f"Starting training from epoch {start_epoch + 1}/{config.num_epochs}")
    print(f"{'=' * 70}\n")

    try:
        for epoch in range(start_epoch, config.num_epochs):
            epoch_start = time.time()

            # Margin warmup
            if epoch < config.margin_warmup_epochs:
                current_margin = config.arcface_margin_start + (
                    config.arcface_margin_end - config.arcface_margin_start
                ) * (epoch / max(1, config.margin_warmup_epochs))
            else:
                current_margin = config.arcface_margin_end

            if hasattr(model, "head") and hasattr(model.head, "set_margin"):
                model.head.set_margin(current_margin)

            # Log current LR (multi-step: readable per-epoch value)
            current_lr = get_lr(global_step, epoch, config)
            print(f"Epoch {epoch + 1}/{config.num_epochs}  LR={current_lr:.2e}  margin={current_margin:.3f}")

            # Train
            train_loss, train_acc, global_step = train_epoch(
                model, train_loader, optimizer, scaler,
                epoch, global_step, config, device, amp_device_type,
                config.checkpoint_dir,
            )

            # Classification validation
            val_loss, val_acc = 0.0, 0.0
            if has_val:
                try:
                    val_loss, val_acc = validate(model, val_loader, config, device, amp_device_type)
                except Exception as e:
                    print(f"  Validation error: {e}")

            # Embedding validation (every 2 epochs)
            embed_acc = 0.0
            if has_val and (epoch + 1) % 2 == 0:
                try:
                    embed_metrics = validate_embeddings(
                        model, val_loader, config, device, amp_device_type
                    )
                    embed_acc = embed_metrics["verification_accuracy"]
                    print(
                        f"  Embed Acc: {embed_acc:.2f}%  "
                        f"TAR@FAR=1%: {embed_metrics['tar_at_far001']:.2f}%  "
                        f"AUC: {embed_metrics['auc']:.4f}"
                    )
                except Exception as e:
                    print(f"  Embedding validation error: {e}")

            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch + 1}/{config.num_epochs} | {epoch_time / 60:.1f} min | margin={current_margin:.3f}")
            print(f"  Train -> Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            if has_val:
                print(f"  Val   -> Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["embed_acc"].append(embed_acc)

            # Save checkpoints
            save_checkpoint(
                config.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth",
                model, optimizer, scaler, epoch + 1, global_step, train_loss,
            )
            save_checkpoint(
                config.checkpoint_dir / "checkpoint_latest.pth",
                model, optimizer, scaler, epoch + 1, global_step, train_loss,
                verbose=False,
            )

            # Best model tracking (priority: embed_acc > val_acc > train_acc)
            improved = False
            if embed_acc > best_embed_acc and embed_acc > 0:
                best_embed_acc = embed_acc
                improved = True
            if has_val and val_acc > best_val_acc:
                best_val_acc = val_acc
                improved = True
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                if not improved and (not has_val or best_val_acc < 1.0) and best_embed_acc < 1.0:
                    improved = True

            if improved:
                if hasattr(model, "backbone"):
                    torch.save(model.backbone.state_dict(), config.checkpoint_dir / "best_backbone.pth")
                    print(f"  Saved best_backbone.pth")
                else:
                    torch.save(model.state_dict(), config.checkpoint_dir / "best_model.pth")

    except KeyboardInterrupt:
        print(f"\n\nTraining interrupted at epoch {epoch + 1}/{config.num_epochs}")
        print(f"Progress saved at step {global_step:,}")
    except Exception as e:
        print(f"\nTraining error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Best train acc:  {best_train_acc:.2f}%")
    if best_val_acc > 0:
        print(f"  Best val acc:    {best_val_acc:.2f}%")
    if best_embed_acc > 0:
        print(f"  Best embed acc:  {best_embed_acc:.2f}%")
    print(f"  Total steps:     {global_step:,}")
    print(f"  Checkpoints:     {config.checkpoint_dir}")

    # Final embedding evaluation
    if has_val:
        print(f"\nFinal Embedding Evaluation (5000 pairs)...")
        try:
            final = validate_embeddings(model, val_loader, config, device, amp_device_type, num_pairs=5000)
            print(f"  Verification Acc: {final['verification_accuracy']:.2f}% @ threshold={final['best_threshold']:.3f}")
            print(f"  TAR@FAR=1%: {final['tar_at_far001']:.2f}%")
            print(f"  AUC: {final['auc']:.4f}")
            print(f"  Pos sim: {final['pos_sim_mean']:.3f} +/- {final['pos_sim_std']:.3f}")
            print(f"  Neg sim: {final['neg_sim_mean']:.3f} +/- {final['neg_sim_std']:.3f}")
        except Exception as e:
            print(f"  Final evaluation error: {e}")

    # Save training curves
    try:
        _save_training_curves(history, config.checkpoint_dir)
    except Exception:
        pass

    return history


def _save_training_curves(history, save_dir):
    """Save training curves plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history["train_loss"]:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = list(range(1, len(history["train_loss"]) + 1))

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train", linewidth=2, markersize=4)
    if any(history["val_loss"]):
        axes[0].plot(epochs, history["val_loss"], "r-s", label="Val", linewidth=2, markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train", linewidth=2, markersize=4)
    if any(history["val_acc"]):
        axes[1].plot(epochs, history["val_acc"], "r-s", label="Val", linewidth=2, markersize=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved to {save_dir / 'training_curves.png'}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train MobileFaceNet + ArcFace")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Base learning rate (default: 0.1, ArcFace standard)")
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--scale", type=float, default=64.0, help="ArcFace scale")
    parser.add_argument("--margin", type=float, default=0.5, help="ArcFace margin")
    parser.add_argument("--margin-warmup", type=int, default=5, help="Margin warmup epochs")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--train-dir", type=str, default=None)
    parser.add_argument("--val-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint (sets auto_resume=True)")
    parser.add_argument("--lr-schedule", type=str, default="multistep",
                        choices=["multistep", "cosine"],
                        help="LR schedule type (default: multistep)")
    parser.add_argument("--milestones", type=int, nargs="+", default=None,
                        help="Epoch milestones for multistep LR, e.g. --milestones 10 18 22")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="Warmup steps for cosine LR (default: 1000)")
    args = parser.parse_args()

    config = TrainingConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.base_lr = args.lr
    config.embedding_dim = args.embedding_dim
    config.arcface_scale = args.scale
    config.arcface_margin = args.margin
    config.arcface_margin_end = args.margin
    config.margin_warmup_epochs = args.margin_warmup
    config.grad_clip = args.grad_clip
    config.checkpoint_every_n_steps = args.checkpoint_every
    config.lr_schedule = args.lr_schedule

    if args.milestones is not None:
        config.lr_milestones = args.milestones
    if args.warmup_steps is not None:
        config.warmup_steps = args.warmup_steps
    if args.workers is not None:
        config.num_workers = args.workers
    if args.train_dir:
        config.train_dir = Path(args.train_dir)
    if args.val_dir:
        config.val_dir = Path(args.val_dir)
    if args.checkpoint_dir:
        config.checkpoint_dir = Path(args.checkpoint_dir)
    if args.no_amp:
        config.use_amp = False

    # Bug #1 Fix: --resume flag now actually enables auto_resume
    if args.resume:
        config.auto_resume = True

    train(config)


if __name__ == "__main__":
    main()
