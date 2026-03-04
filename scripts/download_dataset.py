"""
Download VGGFace2-112x112 Dataset and Create Train/Val Split

Downloads the pre-aligned VGGFace2 dataset from Kaggle using kagglehub
and creates a 90/10 train/validation split at the identity level.

Dataset: yakhyokhuja/vggface2-112x112
  - 8,631 identities, ~3.14 million images
  - Pre-aligned 112x112 RGB faces
  - ~18.69 GB total

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --test-size 0.1 --seed 42

Requirements:
    pip install kagglehub scikit-learn

Notes:
    - Requires Kaggle API authentication (~/.kaggle/kaggle.json)
    - First download takes 10-15 minutes (18.69 GB)
    - Subsequent runs use cached data
    - Train/val split is deterministic with fixed seed
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment detection (matches config.py pattern)
# ---------------------------------------------------------------------------
try:
    from google.colab import drive  # noqa: F401

    IN_COLAB = True
    PROJECT_ROOT = Path("/content/face_recognition")
except ImportError:
    IN_COLAB = False
    PROJECT_ROOT = Path(__file__).resolve().parent.parent


def download_dataset():
    """Download VGGFace2-112x112 from Kaggle via kagglehub."""
    import kagglehub

    DATASET_HANDLE = "yakhyokhuja/vggface2-112x112"

    print(f"{'=' * 70}")
    print(f"VGGFace2-112x112 Dataset Download")
    print(f"{'=' * 70}")
    print(f"  Handle:     {DATASET_HANDLE}")
    print(f"  Identities: 8,631")
    print(f"  Images:     ~3.14 million")
    print(f"  Format:     112x112 RGB (pre-aligned)")
    print(f"  Size:       ~18.69 GB")
    print(f"{'=' * 70}")
    print(f"\nDownloading (cached if already downloaded)...")

    download_path = kagglehub.dataset_download(DATASET_HANDLE)
    print(f"Download path: {download_path}")

    # Show contents
    contents = os.listdir(download_path)
    print(f"\nDownloaded contents:")
    for item in sorted(contents):
        full_path = os.path.join(download_path, item)
        if os.path.isdir(full_path):
            try:
                sub_count = len(os.listdir(full_path))
                print(f"  {item}/ ({sub_count} items)")
            except OSError:
                print(f"  {item}/")
        else:
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            print(f"  {item} ({size_mb:.1f} MB)")

    return download_path


def find_vggface2_dir(download_path):
    """Locate the directory containing identity folders (id_*)."""
    for loc in [os.path.join(download_path, "vggface2_112x112"), download_path]:
        if os.path.exists(loc):
            subdirs = [
                d for d in os.listdir(loc)
                if os.path.isdir(os.path.join(loc, d)) and d.startswith("id_")
            ]
            if subdirs:
                print(f"Found VGGFace2 directory: {loc}")
                print(f"  {len(subdirs)} identity directories")
                return loc

    raise FileNotFoundError(
        f"Could not find vggface2_112x112 directory with id_* folders "
        f"in {download_path}.\nContents: {os.listdir(download_path)}"
    )


def create_train_val_split(vggface2_dir, data_dir, test_size=0.10, seed=42):
    """
    Split dataset into train/val at the identity level.

    Uses symlinks where possible (fast), falls back to copying.
    Handles single-sample classes gracefully for stratified splitting.
    """
    from sklearn.model_selection import train_test_split

    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "val"

    # Get identity directories
    identity_dirs = sorted([
        d for d in os.listdir(vggface2_dir)
        if os.path.isdir(os.path.join(vggface2_dir, d)) and d.startswith("id_")
    ])
    print(f"\nTotal identities: {len(identity_dirs)}")

    # Split
    train_ids, val_ids = train_test_split(
        identity_dirs,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )
    print(f"Train identities: {len(train_ids)}")
    print(f"Val identities:   {len(val_ids)}")

    # Clean existing directories
    start_time = time.time()
    for dir_path in [train_dir, val_dir]:
        if dir_path.exists():
            print(f"Removing existing {dir_path.name}/ ...")
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

    # Setup directories
    for split_name, id_list, target_dir in [
        ("train", train_ids, train_dir),
        ("val", val_ids, val_dir),
    ]:
        print(f"\nSetting up {split_name} ({len(id_list)} identities)...")
        success = 0
        use_symlinks = True

        for idx, identity_id in enumerate(id_list):
            src = os.path.join(vggface2_dir, identity_id)
            dst = os.path.join(target_dir, identity_id)

            try:
                if use_symlinks:
                    try:
                        os.symlink(src, dst, target_is_directory=True)
                        success += 1
                    except (OSError, NotImplementedError):
                        if idx == 0:
                            print("  Symlinks not supported, copying instead...")
                            use_symlinks = False
                        shutil.copytree(src, dst)
                        success += 1
                else:
                    shutil.copytree(src, dst)
                    success += 1
            except Exception as e:
                print(f"  Warning: {identity_id}: {e}")

            if (idx + 1) % 1000 == 0:
                print(f"  ... {idx + 1}/{len(id_list)} processed")

        method = "linked" if use_symlinks else "copied"
        print(f"  {success} identities {method}")

    elapsed = time.time() - start_time
    print(f"\nSplit complete in {elapsed:.1f}s")
    print(f"  Train: {train_dir}")
    print(f"  Val:   {val_dir}")

    return train_dir, val_dir


def count_dataset_stats(data_dir, split_name):
    """Count identities and images in a dataset split."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"  {split_name}: directory not found")
        return 0, 0

    identities = [d for d in data_dir.iterdir() if d.is_dir()]
    total_images = 0

    for idx, identity_dir in enumerate(identities):
        images = (
            list(identity_dir.glob("*.jpg"))
            + list(identity_dir.glob("*.jpeg"))
            + list(identity_dir.glob("*.png"))
        )
        total_images += len(images)
        if (idx + 1) % 1000 == 0:
            print(f"    ... {idx + 1}/{len(identities)} identities counted")

    return len(identities), total_images


def main():
    parser = argparse.ArgumentParser(description="Download VGGFace2-112x112 and create train/val split")
    parser.add_argument("--test-size", type=float, default=0.10, help="Validation split ratio (default: 0.10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory (default: PROJECT_ROOT/data)")
    parser.add_argument("--stats-only", action="store_true", help="Only show dataset statistics, don't download")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data"

    if args.stats_only:
        print(f"\nDataset Statistics")
        print(f"{'=' * 50}")
        for split in ["train", "val"]:
            split_dir = data_dir / split
            n_ids, n_imgs = count_dataset_stats(split_dir, split)
            print(f"  {split.capitalize()}:")
            print(f"    Identities: {n_ids:,}")
            print(f"    Images:     {n_imgs:,}")
            if n_ids > 0:
                print(f"    Avg/id:     {n_imgs / n_ids:.1f}")
        return

    # Step 1: Download
    download_path = download_dataset()

    # Step 2: Find dataset directory
    vggface2_dir = find_vggface2_dir(download_path)

    # Step 3: Create split
    train_dir, val_dir = create_train_val_split(
        vggface2_dir, data_dir,
        test_size=args.test_size,
        seed=args.seed,
    )

    # Step 4: Verify
    print(f"\n{'=' * 50}")
    print(f"Dataset Statistics")
    print(f"{'=' * 50}")

    for split_name, split_dir in [("Train", train_dir), ("Val", val_dir)]:
        n_ids, n_imgs = count_dataset_stats(split_dir, split_name)
        print(f"  {split_name}:")
        print(f"    Identities: {n_ids:,}")
        print(f"    Images:     {n_imgs:,}")
        if n_ids > 0:
            print(f"    Avg/id:     {n_imgs / n_ids:.1f}")

    print(f"\nDataset ready for training!")
    print(f"  Split is deterministic (seed={args.seed})")


if __name__ == "__main__":
    main()
