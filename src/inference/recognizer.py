"""Face Recognition Inference Module.

This module provides high-level face recognition capabilities:
    - Face embedding extraction
    - Face identification (1:N matching)
    - Face verification (1:1 matching)
    - ONNX runtime support for fast inference

Usage:
    >>> recognizer = FaceRecognizer('models/best_model.pth')
    >>> embedding = recognizer.get_embedding(image)
    >>> match = recognizer.identify(image, database)
    >>> is_same = recognizer.verify(image1, image2)
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple, Union, Dict

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    """Face Recognition Inference Module"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import Optional, Union, List, Tuple, Dict
import onnxruntime as ort


class FaceRecognizer:
    def __init__(self, model_path: str, device: str = 'cuda', use_onnx: bool = False):
        self.device = device
        self.use_onnx = use_onnx
        if use_onnx:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'cuda' in device else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
        else:
            checkpoint = torch.load(model_path, map_location=device)
            from src.models.mobile_arcface import MobileFaceNet
            self.model = MobileFaceNet(emb=128)
            self.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
            self.model.to(device).eval()
    
    def preprocess(self, image):
        if isinstance(image, Image.Image): image = np.array(image)
        if image.shape[:2] != (112, 112): image = cv2.resize(image, (112, 112))
        image = (image.astype(np.float32) / 255.0 - 0.5) / 0.5
        if len(image.shape) == 3: image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image).unsqueeze(0)
    
    @torch.no_grad()
    def get_embedding(self, image):
        if not isinstance(image, torch.Tensor): image = self.preprocess(image)
        if self.use_onnx: return self.session.run(None, {self.input_name: image.numpy()})[0].flatten()
        return self.model(image.to(self.device)).cpu().numpy().flatten()
    
    @torch.no_grad()
    def get_embeddings_batch(self, images):
        batch = torch.cat([self.preprocess(img) for img in images], dim=0)
        if self.use_onnx: return self.session.run(None, {self.input_name: batch.numpy()})[0]
        return self.model(batch.to(self.device)).cpu().numpy()
    
    def compare(self, emb1, emb2):
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    
    def recognize(self, image, embedding_db, threshold=0.4):
        emb = self.get_embedding(image)
        person_id, sim = embedding_db.search(emb)
        return (person_id, sim) if sim >= threshold else (None, sim)
    
    def verify(self, image1, image2, threshold=0.4):
        sim = self.compare(self.get_embedding(image1), self.get_embedding(image2))
        return sim >= threshold, sim


class EmbeddingDatabase:
    def __init__(self): self.embeddings = {}
    def add(self, person_id, emb): self.embeddings[person_id] = emb / np.linalg.norm(emb)
    def remove(self, person_id): self.embeddings.pop(person_id, None)
    def search(self, query):
        if not self.embeddings: return None, 0.0
        query = query / np.linalg.norm(query)
        return max(((pid, float(np.dot(query, e))) for pid, e in self.embeddings.items()), key=lambda x: x[1], default=(None, 0.0))
    def save(self, path):
        import pickle
        with open(path, 'wb') as f: pickle.dump(self.embeddings, f)
    def load(self, path):
        import pickle
        with open(path, 'rb') as f: self.embeddings = pickle.load(f)
    def __len__(self): return len(self.embeddings)
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class FaceRecognizer:
    """Face Recognition Engine.
    
    Provides face recognition using either PyTorch or ONNX runtime.
    ONNX is recommended for production as it's 2-3x faster.
    
    Args:
        model_path: Path to model file (.pth or .onnx)
        device: Device for inference ('cuda', 'cpu', or 'auto')
        use_onnx: Force ONNX runtime (default: auto-detect from file)
        
    Example:
        >>> recognizer = FaceRecognizer('models/mobilefacenet.onnx')
        >>> 
        >>> # Get embedding
        >>> image = cv2.imread('face.jpg')
        >>> embedding = recognizer.get_embedding(image)
        >>> 
        >>> # Verify two faces
        >>> is_same, score = recognizer.verify(image1, image2)
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'auto',
        use_onnx: Optional[bool] = None,
    ):
        self.model_path = Path(model_path)
        
        # Determine if using ONNX
        if use_onnx is None:
            use_onnx = self.model_path.suffix.lower() == '.onnx'
        self.use_onnx = use_onnx
        
        # Setup device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Load model
        if self.use_onnx:
            self._load_onnx_model()
        else:
            self._load_pytorch_model()
        
        # Preprocessing parameters
        self.input_size = (112, 112)
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        
        print(f"FaceRecognizer loaded: {self.model_path.name}")
        print(f"  Backend: {'ONNX' if self.use_onnx else 'PyTorch'}")
        print(f"  Device: {self.device}")
    
    def _load_pytorch_model(self) -> None:
        """Load PyTorch model."""
        from ..models import MobileFaceNet
        
        # Try to load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Determine embedding size from checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Find embedding size from state dict
        embedding_key = 'backbone.embedding.0.weight'
        if embedding_key in state_dict:
            embedding_size = state_dict[embedding_key].shape[0]
        else:
            embedding_size = 128  # Default
        
        # Create model
        self.model = MobileFaceNet(embedding_size=embedding_size)
        
        # Load weights (handle both backbone-only and full model)
        try:
            # Try loading backbone weights only
            backbone_dict = {k.replace('backbone.', ''): v 
                           for k, v in state_dict.items() 
                           if k.startswith('backbone.')}
            if backbone_dict:
                self.model.load_state_dict(backbone_dict, strict=False)
            else:
                self.model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Warning: Could not load all weights: {e}")
            self.model.load_state_dict(state_dict, strict=False)
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _load_onnx_model(self) -> None:
        """Load ONNX model with ONNX Runtime."""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not available. Install with: pip install onnxruntime-gpu")
        
        # Setup ONNX session
        providers = []
        if 'cuda' in str(self.device):
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        self.onnx_session = ort.InferenceSession(
            str(self.model_path),
            providers=providers,
        )
        
        # Get input/output names
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_name = self.onnx_session.get_outputs()[0].name
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference.
        
        Args:
            image: Input image (H, W, 3) in BGR or RGB format
            
        Returns:
            Preprocessed array (1, 3, 112, 112)
        """
        # Resize if needed
        if image.shape[:2] != self.input_size:
            image = cv2.resize(image, self.input_size)
        
        # Ensure RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        # Assume already RGB if 3 channels
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        
        # Transpose to (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    @torch.no_grad()
    def get_embedding(
        self, 
        image: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """Extract face embedding from image.
        
        Args:
            image: Face image (H, W, 3), should be aligned 112x112
            normalize: L2-normalize the embedding
            
        Returns:
            Face embedding (embedding_size,)
        """
        # Preprocess
        preprocessed = self.preprocess(image)
        
        if self.use_onnx:
            # ONNX inference
            outputs = self.onnx_session.run(
                [self.output_name],
                {self.input_name: preprocessed},
            )
            embedding = outputs[0][0]
        else:
            # PyTorch inference
            x = torch.from_numpy(preprocessed).to(self.device)
            embedding = self.model(x).cpu().numpy()[0]
        
        # Normalize
        if normalize:
            embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
        
        return embedding
    
    @torch.no_grad()
    def get_embeddings_batch(
        self,
        images: List[np.ndarray],
        normalize: bool = True,
    ) -> np.ndarray:
        """Extract embeddings from multiple images.
        
        Args:
            images: List of face images
            normalize: L2-normalize embeddings
            
        Returns:
            Embeddings array (N, embedding_size)
        """
        if not images:
            return np.array([])
        
        # Preprocess all images
        preprocessed = np.concatenate(
            [self.preprocess(img) for img in images],
            axis=0
        )
        
        if self.use_onnx:
            outputs = self.onnx_session.run(
                [self.output_name],
                {self.input_name: preprocessed},
            )
            embeddings = outputs[0]
        else:
            x = torch.from_numpy(preprocessed).to(self.device)
            embeddings = self.model(x).cpu().numpy()
        
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
            embeddings = embeddings / norms
        
        return embeddings
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score (-1 to 1)
        """
        # Normalize
        emb1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-10)
        emb2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-10)
        
        return np.dot(emb1_norm, emb2_norm)
    
    def verify(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        threshold: float = 0.4,
    ) -> Tuple[bool, float]:
        """Verify if two face images are the same person.
        
        Args:
            image1: First face image
            image2: Second face image
            threshold: Similarity threshold for same person
            
        Returns:
            is_same: True if same person
            similarity: Similarity score
        """
        emb1 = self.get_embedding(image1)
        emb2 = self.get_embedding(image2)
        
        similarity = self.compute_similarity(emb1, emb2)
        is_same = similarity >= threshold
        
        return is_same, similarity
    
    def identify(
        self,
        image: np.ndarray,
        database: 'EmbeddingDatabase',
        threshold: float = 0.4,
        top_k: int = 1,
    ) -> List[Tuple[str, float]]:
        """Identify a face against a database.
        
        Args:
            image: Query face image
            database: EmbeddingDatabase instance
            threshold: Minimum similarity threshold
            top_k: Number of top matches to return
            
        Returns:
            List of (person_id, similarity) tuples
        """
        embedding = self.get_embedding(image)
        return database.search(embedding, top_k=top_k, threshold=threshold)


def load_model_for_inference(
    model_path: str,
    device: str = 'auto',
) -> FaceRecognizer:
    """Load a model for inference.
    
    Convenience function to create FaceRecognizer.
    
    Args:
        model_path: Path to model file
        device: Device for inference
        
    Returns:
        FaceRecognizer instance
    """
    return FaceRecognizer(model_path, device=device)


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    embedding_size: int = 128,
    input_size: Tuple[int, int] = (112, 112),
    opset_version: int = 12,
    simplify: bool = True,
) -> None:
    """Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model (MobileFaceNet or MobileArcFaceModel)
        output_path: Path to save ONNX model
        embedding_size: Output embedding size
        input_size: Input image size
        opset_version: ONNX opset version
        simplify: Simplify ONNX graph (requires onnx-simplifier)
    """
    import torch.onnx
    
    # Get backbone if full model
    if hasattr(model, 'backbone'):
        export_model = model.backbone
    else:
        export_model = model
    
    export_model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    
    # Export
    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=['input'],
        output_names=['embedding'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'embedding': {0: 'batch_size'},
        },
    )
    
    print(f"Exported model to: {output_path}")
    
    # Simplify if requested
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            
            model_onnx = onnx.load(output_path)
            model_simplified, check = onnx_simplify(model_onnx)
            
            if check:
                onnx.save(model_simplified, output_path)
                print("  Simplified ONNX graph")
            else:
                print("  Warning: Could not simplify ONNX graph")
        except ImportError:
            print("  Note: Install onnx-simplifier for smaller model: pip install onnx-simplifier")


if __name__ == "__main__":
    print("Testing FaceRecognizer...")
    
    # Test preprocessing
    recognizer = FaceRecognizer.__new__(FaceRecognizer)
    recognizer.input_size = (112, 112)
    recognizer.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    recognizer.std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    
    dummy_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    preprocessed = recognizer.preprocess(dummy_image)
    print(f"Preprocessed shape: {preprocessed.shape}")
    assert preprocessed.shape == (1, 3, 112, 112)
    
    # Test similarity
    emb1 = np.random.randn(128)
    emb2 = np.random.randn(128)
    sim = recognizer.compute_similarity(emb1, emb2)
    print(f"Similarity: {sim:.4f}")
    
    print("\n✓ Recognizer tests passed!")
