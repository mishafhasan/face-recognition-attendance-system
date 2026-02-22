"""
Configuration file for Face-Based Attendance System
Supports both Google Colab and local environments
"""

import os
from pathlib import Path

# Detect environment
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Set project root based on environment
if IN_COLAB:
    PROJECT_ROOT = "/content/drive/MyDrive/face_based_attendance_system"
else:
    # For local environment, use the directory containing this file
    PROJECT_ROOT = str(Path(__file__).parent.resolve())

# Directory structure
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")

# Data subdirectories
VGGFACE2_DIR = os.path.join(DATA_DIR, "vggface2")
LFW_DIR = os.path.join(DATA_DIR, "lfw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
TEST_IMAGES_DIR = os.path.join(DATA_DIR, "test_images")
ENROLLMENTS_DIR = os.path.join(DATA_DIR, "enrollments")

# Dataset configurations
DATASETS = {
    'vggface2': {
        'path': VGGFACE2_DIR,
        'kaggle_handle': 'yakhyokhuja/vggface2-112x112',
        'description': 'VGGFace2 112x112 - Pre-aligned face recognition dataset (8,631 identities, ~3.14M images)',
        'structure': 'data/nXXXXXX/*.jpg',
        'use_for': 'training + validation (auto 90/10 split)',
    },
    'lfw': {
        'path': LFW_DIR,
        'source': 'sklearn.datasets.fetch_lfw_pairs',
        'description': 'LFW Pairs - Labeled Faces in the Wild (evaluation benchmark)',
        'use_for': 'evaluation (pair verification)',
    },
}

# Model subdirectories
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")

# Model hyperparameters
IMG_SIZE = 112
EMBEDDING_SIZE = 512
ARCFACE_SCALE = 64.0
ARCFACE_MARGIN = 0.5

# Training settings
BATCH_SIZE = 256
NUM_WORKERS = 4
PIN_MEMORY = True
VAL_SPLIT = 0.1

def ensure_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_DIR, MODELS_DIR, VGGFACE2_DIR, LFW_DIR, PROCESSED_DIR,
        TEST_IMAGES_DIR, ENROLLMENTS_DIR, CHECKPOINTS_DIR
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print(f"✅ All directories created/verified")

def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("FACE-BASED ATTENDANCE SYSTEM - CONFIGURATION")
    print("=" * 60)
    print(f"Environment: {'Google Colab' if IN_COLAB else 'Local'}")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"\nDirectories:")
    print(f"  Data: {DATA_DIR}")
    print(f"  Models: {MODELS_DIR}")
    print(f"  Backend: {BACKEND_DIR}")
    print(f"  Notebooks: {NOTEBOOKS_DIR}")
    print(f"\nModel Settings:")
    print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Embedding Size: {EMBEDDING_SIZE}")
    print(f"  ArcFace Scale: {ARCFACE_SCALE}")
    print(f"  ArcFace Margin: {ARCFACE_MARGIN}")
    print(f"\nTraining Settings:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Num Workers: {NUM_WORKERS}")
    print(f"  Val Split: {VAL_SPLIT}")
    print("=" * 60)

if __name__ == "__main__":
    ensure_directories()
    print_config()
