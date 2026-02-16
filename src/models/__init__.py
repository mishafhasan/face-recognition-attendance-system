"""Neural network models for face recognition.

This module contains:
    - MobileFaceNet: Lightweight face recognition backbone
    - ArcFace: Angular margin loss for metric learning
    - CosFace: Cosine margin loss alternative
    - FaceDetector: MTCNN-based face detection wrapper
"""

from .mobile_arcface import (
    ConvBlock,
    DepthWise,
    DepthWiseResidual,
    MobileFaceNet,
    MobileArcFaceModel,
)
from .losses import ArcFace, CosFace
from .face_detector import FaceDetector

__all__ = [
    "ConvBlock",
    "DepthWise",
    "DepthWiseResidual",
    "MobileFaceNet",
    "MobileArcFaceModel",
    "ArcFace",
    "CosFace",
    "FaceDetector",
]
