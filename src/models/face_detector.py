"""Face Detection and Alignment Module.

This module provides face detection and alignment using MTCNN.
The detected faces are aligned to a canonical 112x112 format
suitable for the MobileFaceNet backbone.

Features:
    - Multi-scale face detection
    - 5-point facial landmark detection
    - Face alignment using similarity transform
    - Batch processing support

Reference:
    Zhang et al., "Joint Face Detection and Alignment using Multi-task 
    Cascaded Convolutional Networks", IEEE Signal Processing Letters 2016
"""

import math
from typing import Optional, Tuple, List, Union

import numpy as np
import cv2
import torch
from PIL import Image
from typing import Optional, Tuple, List, Union

# Try to import facenet_pytorch MTCNN
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("Warning: facenet_pytorch not installed. Install with: pip install facenet-pytorch")

ARCFACE_LANDMARKS = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


class FaceDetector:
    def __init__(self, device='cpu', min_face_size=20, thresholds=[0.6, 0.7, 0.7]):
        self.device = device
        self.mtcnn = MTCNN(image_size=112, margin=0, min_face_size=min_face_size, thresholds=thresholds, factor=0.709, post_process=False, device=device, keep_all=True)
    
    def detect(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return self.mtcnn.detect(image, landmarks=True)
    
    def detect_and_align(self, image, return_all=False):
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image_pil = image
            image = np.array(image)[:, :, ::-1]
        
        boxes, probs, landmarks = self.detect(image_pil)
        if boxes is None: return [] if return_all else None
        
        aligned = []
        for i in range(len(boxes)):
            face = align_face(image, landmarks[i]) if landmarks is not None and landmarks[i] is not None else cv2.resize(image[max(0,int(boxes[i][1])):int(boxes[i][3]), max(0,int(boxes[i][0])):int(boxes[i][2])], (112, 112))
            aligned.append(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        
        if return_all: return aligned
        if len(aligned) == 1: return aligned[0]
        areas = [(boxes[i][2]-boxes[i][0]) * (boxes[i][3]-boxes[i][1]) for i in range(len(boxes))]
        return aligned[np.argmax(areas)]


def align_face(image, landmarks, output_size=112):
    src = landmarks.astype(np.float32)
    M = cv2.estimateAffinePartial2D(src, ARCFACE_LANDMARKS)[0]
    if M is None: M = cv2.getAffineTransform(src[:3], ARCFACE_LANDMARKS[:3])
    return cv2.warpAffine(image, M, (output_size, output_size), borderMode=cv2.BORDER_REPLICATE)


def preprocess_face(face, mean=0.5, std=0.5):
    face = face.astype(np.float32) / 255.0
    face = (face - mean) / std
    return torch.from_numpy(np.transpose(face, (2, 0, 1))).unsqueeze(0)


def batch_preprocess_faces(faces, mean=0.5, std=0.5):
    return torch.stack([preprocess_face(f, mean, std).squeeze(0) for f in faces])


# Standard face alignment template for 112x112 images
# These are the reference landmark positions for aligned faces
ARCFACE_REFERENCE_LANDMARKS = np.array([
    [38.2946, 51.6963],   # Left eye
    [73.5318, 51.5014],   # Right eye
    [56.0252, 71.7366],   # Nose tip
    [41.5493, 92.3655],   # Left mouth corner
    [70.7299, 92.2041],   # Right mouth corner
], dtype=np.float32)


def estimate_similarity_transform(
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> np.ndarray:
    """Estimate similarity transformation matrix.
    
    Computes the optimal similarity transform (rotation, scale, translation)
    that maps source points to destination points.
    
    Args:
        src_points: Source landmark points (N, 2)
        dst_points: Destination landmark points (N, 2)
        
    Returns:
        3x3 transformation matrix
    """
    num_points = src_points.shape[0]
    
    # Center the points
    src_mean = np.mean(src_points, axis=0)
    dst_mean = np.mean(dst_points, axis=0)
    
    src_centered = src_points - src_mean
    dst_centered = dst_points - dst_mean
    
    # Compute scale and rotation
    src_std = np.std(src_centered)
    dst_std = np.std(dst_centered)
    
    src_norm = src_centered / (src_std + 1e-8)
    dst_norm = dst_centered / (dst_std + 1e-8)
    
    # Compute rotation using SVD
    H = src_norm.T @ dst_norm
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (no reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Scale
    scale = dst_std / (src_std + 1e-8)
    
    # Build transformation matrix
    T = np.eye(3, dtype=np.float32)
    T[:2, :2] = scale * R
    T[:2, 2] = dst_mean - scale * R @ src_mean
    
    return T


def align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    target_size: Tuple[int, int] = (112, 112),
    reference_landmarks: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Align face image using facial landmarks.
    
    Applies a similarity transform to align the face to a canonical
    position based on reference landmark positions.
    
    Args:
        image: Input image (H, W, 3) in RGB format
        landmarks: 5-point facial landmarks (5, 2)
        target_size: Output image size (default: 112x112)
        reference_landmarks: Reference landmark positions (optional)
        
    Returns:
        Aligned face image (target_size[0], target_size[1], 3)
    """
    if reference_landmarks is None:
        reference_landmarks = ARCFACE_REFERENCE_LANDMARKS
    
    # Ensure landmarks are float32
    landmarks = np.array(landmarks, dtype=np.float32)
    
    # Estimate transformation
    transform = estimate_similarity_transform(landmarks, reference_landmarks)
    
    # Apply affine transform
    aligned = cv2.warpAffine(
        image,
        transform[:2, :],
        target_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    
    return aligned


class FaceDetector:
    """Face Detection and Alignment using MTCNN.
    
    Provides face detection with facial landmark localization and
    automatic face alignment for recognition.
    
    Args:
        device: Device to run detection on ('cuda' or 'cpu')
        min_face_size: Minimum face size in pixels (default: 20)
        thresholds: MTCNN stage thresholds (default: [0.6, 0.7, 0.7])
        keep_all: Return all detected faces (default: True)
        target_size: Output aligned face size (default: (112, 112))
        
    Example:
        >>> detector = FaceDetector(device='cuda')
        >>> image = cv2.imread('photo.jpg')
        >>> faces, boxes, landmarks = detector.detect_and_align(image)
        >>> for face in faces:
        ...     # face is aligned 112x112 image ready for recognition
        ...     embedding = model(preprocess(face))
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        min_face_size: int = 20,
        thresholds: List[float] = None,
        keep_all: bool = True,
        target_size: Tuple[int, int] = (112, 112),
    ):
        if not MTCNN_AVAILABLE:
            raise ImportError(
                "facenet_pytorch is required for face detection. "
                "Install with: pip install facenet-pytorch"
            )
        
        if thresholds is None:
            thresholds = [0.6, 0.7, 0.7]
        
        self.device = device
        self.target_size = target_size
        self.keep_all = keep_all
        
        # Initialize MTCNN
        self.mtcnn = MTCNN(
            image_size=160,  # Internal processing size
            margin=0,
            min_face_size=min_face_size,
            thresholds=thresholds,
            factor=0.709,
            post_process=False,  # We'll do our own preprocessing
            device=device,
            keep_all=keep_all,
        )
    
    def detect(
        self, 
        image: Union[np.ndarray, Image.Image]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect faces in image.
        
        Args:
            image: Input image (H, W, 3) RGB numpy array or PIL Image
            
        Returns:
            boxes: Bounding boxes (N, 4) as [x1, y1, x2, y2] or None
            probs: Detection probabilities (N,) or None
            landmarks: Facial landmarks (N, 5, 2) or None
        """
        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
        
        # Detect faces
        boxes, probs, landmarks = self.mtcnn.detect(image_pil, landmarks=True)
        
        return boxes, probs, landmarks
    
    def detect_and_align(
        self,
        image: Union[np.ndarray, Image.Image],
        return_largest: bool = False,
    ) -> Tuple[List[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect and align faces in image.
        
        Args:
            image: Input image (H, W, 3) RGB numpy array or PIL Image
            return_largest: Only return the largest face (default: False)
            
        Returns:
            aligned_faces: List of aligned face images (112, 112, 3)
            boxes: Bounding boxes (N, 4) or None
            landmarks: Facial landmarks (N, 5, 2) or None
        """
        # Ensure numpy format for alignment
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Detect faces
        boxes, probs, landmarks = self.detect(image_np)
        
        if boxes is None:
            return [], None, None
        
        # Handle single face detection
        if len(boxes.shape) == 1:
            boxes = boxes.reshape(1, -1)
            landmarks = landmarks.reshape(1, 5, 2)
            probs = np.array([probs])
        
        # Align faces
        aligned_faces = []
        for i, landmark in enumerate(landmarks):
            aligned = align_face(image_np, landmark, self.target_size)
            aligned_faces.append(aligned)
        
        if return_largest and len(aligned_faces) > 0:
            # Find largest face by box area
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            largest_idx = np.argmax(areas)
            aligned_faces = [aligned_faces[largest_idx]]
            boxes = boxes[largest_idx:largest_idx+1]
            landmarks = landmarks[largest_idx:largest_idx+1]
        
        return aligned_faces, boxes, landmarks
    
    def extract_face(
        self,
        image: Union[np.ndarray, Image.Image],
    ) -> Optional[np.ndarray]:
        """Extract and align the largest face from image.
        
        Convenience method for single face extraction.
        
        Args:
            image: Input image
            
        Returns:
            Aligned face image (112, 112, 3) or None if no face detected
        """
        faces, _, _ = self.detect_and_align(image, return_largest=True)
        if len(faces) > 0:
            return faces[0]
        return None
    
    @torch.no_grad()
    def batch_extract(
        self,
        images: List[Union[np.ndarray, Image.Image]],
    ) -> List[Optional[np.ndarray]]:
        """Extract faces from multiple images.
        
        Args:
            images: List of input images
            
        Returns:
            List of aligned faces (None for images with no face)
        """
        results = []
        for image in images:
            face = self.extract_face(image)
            results.append(face)
        return results


def preprocess_face(
    face: np.ndarray,
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> torch.Tensor:
    """Preprocess aligned face for model input.
    
    Converts aligned face image to normalized tensor.
    
    Args:
        face: Aligned face image (112, 112, 3) in RGB, uint8
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Preprocessed tensor (1, 3, 112, 112) ready for model
    """
    # Convert to float and normalize to [0, 1]
    face = face.astype(np.float32) / 255.0
    
    # Normalize
    face = (face - mean) / std
    
    # Convert to tensor (H, W, C) -> (C, H, W)
    face = np.transpose(face, (2, 0, 1))
    
    # Add batch dimension
    face = torch.from_numpy(face).unsqueeze(0)
    
    return face


def batch_preprocess_faces(
    faces: List[np.ndarray],
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> torch.Tensor:
    """Preprocess multiple aligned faces.
    
    Args:
        faces: List of aligned face images
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Batch tensor (B, 3, 112, 112)
    """
    batch = []
    for face in faces:
        preprocessed = preprocess_face(face, mean, std)
        batch.append(preprocessed)
    
    return torch.cat(batch, dim=0)


if __name__ == "__main__":
    print("Testing Face Detector...")
    
    # Test alignment function
    print("Testing alignment function...")
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    dummy_landmarks = np.array([
        [80, 100],
        [175, 100],
        [128, 150],
        [90, 200],
        [165, 200],
    ], dtype=np.float32)
    
    aligned = align_face(dummy_image, dummy_landmarks)
    print(f"Aligned face shape: {aligned.shape}")
    assert aligned.shape == (112, 112, 3), "Alignment output shape mismatch"
    
    # Test preprocessing
    print("Testing preprocessing...")
    tensor = preprocess_face(aligned)
    print(f"Preprocessed tensor shape: {tensor.shape}")
    assert tensor.shape == (1, 3, 112, 112), "Preprocessing output shape mismatch"
    
    if MTCNN_AVAILABLE:
        print("Testing FaceDetector...")
        detector = FaceDetector(device='cpu')
        print(f"FaceDetector initialized on CPU")
    else:
        print("Skipping FaceDetector test (facenet_pytorch not installed)")
    
    print("\n✓ Face detector tests passed!")
