"""Face Recognition Inference Module.

This module provides production-ready inference:
    - FaceRecognizer: Face embedding extraction and recognition
    - EmbeddingDatabase: Store and search face embeddings
"""

from .recognizer import FaceRecognizer, load_model_for_inference
from .embedding_db import EmbeddingDatabase

__all__ = [
    "FaceRecognizer",
    "load_model_for_inference",
    "EmbeddingDatabase",
]
