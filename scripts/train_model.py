"""
Model Training Script

Trains the face embedding model on LFW dataset using triplet loss.

Usage:
    python train_model.py
    python train_model.py --epochs 30 --batch-size 32
    python train_model.py --resume models/checkpoint_epoch_10.pth
"""
import argparse
import sys
import os
from pathlib import Path
import json
from datetime import datetime

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.ml.model import FaceEmbeddingModel, TripletLoss, create_model
from backend.app.ml.dataset import load_processed_data, create_data_loaders


# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "lfw" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def train_epoch(
    model: FaceEmbeddingModel,
    loader,
    optimizer,
    loss_fn,
    device: str,
    epoch: int
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for anchor, positive, negative in pbar:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        emb_anchor = model(anchor)
        emb_positive = model(positive)
        emb_negative = model(negative)
        
        # Compute loss
        loss = loss_fn(emb_anchor, emb_positive, emb_negative)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / num_batches


def validate_epoch(
    model: FaceEmbeddingModel,
    loader,
    loss_fn,
    device: str,
    epoch: int
) -> float:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
        for anchor, positive, negative in pbar:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            # Forward pass
            emb_anchor = model(anchor)
            emb_positive = model(positive)
            emb_negative = model(negative)
            
            # Compute loss
            loss = loss_fn(emb_anchor, emb_positive, emb_negative)
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / num_batches


def save_checkpoint(
    model: FaceEmbeddingModel,
    optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    path: Path
):
    """Save model checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "embedding_dim": model.embedding_dim
    }, path)


def load_checkpoint(path: Path, model: FaceEmbeddingModel, optimizer=None):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint.get("val_loss", float("inf"))


def plot_training_history(history: dict, save_path: Path):
    """Plot and save training history."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(history["train_loss"], label="Train Loss", marker='o')
    ax.plot(history["val_loss"], label="Val Loss", marker='s')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Triplet Loss")
    ax.set_title("Training History")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 Training plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train face embedding model")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--margin", type=float, default=0.2, help="Triplet loss margin")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")
    
    args = parser.parse_args()
    
    # Setup device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"\n🖥️  Using device: {device}")
    if use_cuda:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Create directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if data exists
    if not DATA_DIR.exists():
        print(f"❌ Data not found at {DATA_DIR}")
        print("   Run 'python scripts/download_lfw.py' first!")
        return
    
    print("\n" + "=" * 50)
    print("🧠 Face Embedding Model Training")
    print("=" * 50)
    
    # Load data
    print("\n📂 Loading data...")
    data = load_processed_data(DATA_DIR)
    print(f"   Train: {len(data['train']['images'])} images")
    print(f"   Val: {len(data['val']['images'])} images")
    print(f"   Classes: {len(data['names'])} people")
    
    # Create data loaders
    print("\n📦 Creating data loaders...")
    loaders = create_data_loaders(data, batch_size=args.batch_size)
    
    # Create model
    print("\n🏗️  Creating model...")
    model = create_model(
        embedding_dim=args.embedding_dim,
        pretrained=True,
        device=device
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {num_params:,}")
    print(f"   Embedding dim: {args.embedding_dim}")
    
    # Loss and optimizer
    loss_fn = TripletLoss(margin=args.margin)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Resume from checkpoint
    start_epoch = 1
    best_val_loss = float("inf")
    
    if args.resume:
        print(f"\n📥 Resuming from {args.resume}...")
        start_epoch, best_val_loss = load_checkpoint(
            Path(args.resume), model, optimizer
        )
        start_epoch += 1
        print(f"   Starting from epoch {start_epoch}")
    
    # Training history
    history = {"train_loss": [], "val_loss": []}
    
    # Training loop
    print("\n" + "=" * 50)
    print("🚀 Starting training...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Margin: {args.margin}")
    print("=" * 50 + "\n")
    
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            # Train
            train_loss = train_epoch(
                model, loaders["train"], optimizer, loss_fn, device, epoch
            )
            
            # Validate
            val_loss = validate_epoch(
                model, loaders["val"], loss_fn, device, epoch
            )
            
            # Update scheduler
            scheduler.step()
            
            # Record history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            
            # Print summary
            print(f"\n📈 Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, epoch, train_loss, val_loss,
                    MODELS_DIR / "best_model.pth"
                )
                print(f"   💾 New best model saved!")
            
            # Save periodic checkpoint
            if epoch % 5 == 0:
                save_checkpoint(
                    model, optimizer, epoch, train_loss, val_loss,
                    MODELS_DIR / f"checkpoint_epoch_{epoch}.pth"
                )
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    
    # Save final model
    save_checkpoint(
        model, optimizer, epoch, train_loss, val_loss,
        MODELS_DIR / "final_model.pth"
    )
    
    # Plot training history
    if len(history["train_loss"]) > 1:
        plot_training_history(history, MODELS_DIR / "training_history.png")
    
    # Save training config
    config = {
        "epochs_trained": epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "margin": args.margin,
        "embedding_dim": args.embedding_dim,
        "best_val_loss": best_val_loss,
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        "device": device,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(MODELS_DIR / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 50)
    print("✅ Training complete!")
    print("=" * 50)
    print(f"\n📁 Models saved to: {MODELS_DIR}")
    print(f"   • best_model.pth (val_loss: {best_val_loss:.4f})")
    print(f"   • final_model.pth")
    print(f"   • training_history.png")
    print(f"\n👉 NEXT STEP: Run 'python scripts/evaluate_model.py' to evaluate on LFW benchmark")


if __name__ == "__main__":
    main()
