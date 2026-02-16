"""Angular Margin Loss Functions for Face Recognition.

This module implements state-of-the-art angular margin losses:
    - ArcFace: Additive Angular Margin Loss
    - CosFace: Large Margin Cosine Loss
    - FocalLoss: For handling class imbalance
    - CombinedMarginLoss: Combines multiple margin types

These losses enhance feature discriminability by introducing
angular/cosine margins that force the model to learn tighter
intra-class clusters and larger inter-class distances.

References:
    ArcFace: Deng et al., "ArcFace: Additive Angular Margin Loss for 
    Deep Face Recognition", CVPR 2019
    https://arxiv.org/abs/1801.07698
    
    CosFace: Wang et al., "CosFace: Large Margin Cosine Loss for 
    Deep Face Recognition", CVPR 2018
    https://arxiv.org/abs/1801.09414
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    """ArcFace: Additive Angular Margin Loss.
    
    Adds an angular margin penalty to the target class angle,
    making the decision boundary more discriminative.
    
    Formula:
        L = -log(exp(s * cos(theta_y + m)) / 
                 (exp(s * cos(theta_y + m)) + sum_j!=y(exp(s * cos(theta_j)))))
    
    Where:
        - theta_y is the angle between feature and weight of target class
        - m is the angular margin (default: 0.5 rad ≈ 28.6°)
        - s is the scaling factor (default: 64)
    
    Args:
        in_features: Size of input features (embedding dimension)
        out_features: Number of classes
        scale: Scaling factor (default: 64.0)
        margin: Angular margin in radians (default: 0.5)
        easy_margin: Use easy margin formulation (default: False)
        
    Example:
        >>> arcface = ArcFace(512, 10000, scale=64.0, margin=0.5)
        >>> embeddings = F.normalize(torch.randn(32, 512), dim=1)
        >>> labels = torch.randint(0, 10000, (32,))
        >>> logits = arcface(embeddings, labels)
        >>> loss = F.cross_entropy(logits, labels)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale: float = 64.0,
        margin: float = 0.5,
        easy_margin: bool = False,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        
        # Learnable class centers (weight matrix)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Precompute margin values for efficiency
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)  # Threshold for easy margin
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute ArcFace logits.
        
        Args:
            embeddings: L2-normalized face embeddings (B, in_features)
            labels: Ground truth class labels (B,)
            
        Returns:
            Scaled logits with angular margin applied to target class (B, out_features)
        """
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity: cos(theta) = x · w
        cosine = F.linear(embeddings, weight)
        cosine = cosine.clamp(-1, 1)  # Numerical stability
        
        # sin(theta) = sqrt(1 - cos^2(theta))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Apply safe margin
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # When cos(theta) < cos(pi - m), use cos(theta) - m*sin(m)
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # One-hot encoding of labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Apply margin only to ground-truth class
        output = one_hot * phi + (1.0 - one_hot) * cosine
        output *= self.scale
        
        return output
    
    def extra_repr(self) -> str:
        """Extra information for print(model)."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"scale={self.scale}, margin={self.margin}"
        )


class CosFace(nn.Module):
    """CosFace: Large Margin Cosine Loss.
    
    Subtracts a cosine margin from the target class similarity.
    Simpler than ArcFace but still effective.
    
    Formula:
        L = -log(exp(s * (cos(theta_y) - m)) / 
                 (exp(s * (cos(theta_y) - m)) + sum_j!=y(exp(s * cos(theta_j)))))
    
    Args:
        in_features: Size of input features (embedding dimension)
        out_features: Number of classes
        scale: Scaling factor (default: 64.0)
        margin: Cosine margin (default: 0.35)
        
    Example:
        >>> cosface = CosFace(512, 10000, scale=64.0, margin=0.35)
        >>> embeddings = F.normalize(torch.randn(32, 512), dim=1)
        >>> labels = torch.randint(0, 10000, (32,))
        >>> logits = cosface(embeddings, labels)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale: float = 64.0,
        margin: float = 0.35,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute CosFace logits.
        
        Args:
            embeddings: L2-normalized face embeddings (B, in_features)
            labels: Ground truth class labels (B,)
            
        Returns:
            Scaled logits with cosine margin applied to target class (B, out_features)
        """
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity
        cosine = F.linear(embeddings, weight)
        
        # Subtract margin from ground-truth class: cos(theta) - m
        phi = cosine - self.margin
        
        # One-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Apply margin only to target class
        output = one_hot * phi + (1.0 - one_hot) * cosine
        output *= self.scale
        
        return output
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"scale={self.scale}, margin={self.margin}"
        )


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    
    Down-weights well-classified examples and focuses on hard examples.
    Useful when combined with ArcFace/CosFace for imbalanced datasets.
    
    Formula:
        FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
    
    Args:
        gamma: Focusing parameter (default: 2.0)
        alpha: Class balancing weight (optional)
        reduction: Reduction method ('mean', 'sum', 'none')
        
    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
        https://arxiv.org/abs/1708.02002
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            logits: Model output logits (B, num_classes)
            labels: Ground truth labels (B,)
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """Combined ArcFace + Focal Loss.
    
    Combines angular margin loss with focal loss for better
    handling of hard examples and class imbalance.
    
    Args:
        in_features: Embedding dimension
        out_features: Number of classes
        scale: ArcFace scaling factor
        margin: ArcFace angular margin
        gamma: Focal loss gamma
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale: float = 64.0,
        margin: float = 0.5,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.arcface = ArcFace(in_features, out_features, scale, margin)
        self.focal = FocalLoss(gamma)
    
    def forward(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined loss.
        
        Args:
            embeddings: Face embeddings (B, in_features)
            labels: Ground truth labels (B,)
            
        Returns:
            Combined loss value
        """
        logits = self.arcface(embeddings, labels)
        loss = self.focal(logits, labels)
        return loss


class CombinedMarginLoss(nn.Module):
    """Combined Margin Loss (ArcFace + CosFace + SphereFace).
    
    Combines multiple margin types for potentially better performance.
    
    Formula:
        cos(m1 * theta + m2) - m3
    
    Where:
        - m1 is SphereFace multiplicative margin
        - m2 is ArcFace additive angular margin  
        - m3 is CosFace additive cosine margin
    
    Args:
        in_features: Dimension of face embeddings
        out_features: Number of identity classes
        scale: Scaling factor (default: 64.0)
        m1: SphereFace multiplicative margin (default: 1.0)
        m2: ArcFace additive angular margin (default: 0.5)
        m3: CosFace additive cosine margin (default: 0.0)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale: float = 64.0,
        m1: float = 1.0,
        m2: float = 0.5,
        m3: float = 0.0,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Precompute
        self.cos_m2 = math.cos(m2)
        self.sin_m2 = math.sin(m2)
        self.threshold = math.cos(math.pi - m2)
    
    def forward(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined margin loss logits."""
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, weight)
        
        # Apply m1 (SphereFace multiplicative margin)
        if self.m1 != 1.0:
            theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
            theta_m1 = self.m1 * theta
            cosine_m1 = torch.cos(theta_m1)
        else:
            cosine_m1 = cosine
        
        # Apply m2 (ArcFace additive margin)
        sine = torch.sqrt(torch.clamp(1.0 - cosine_m1.pow(2), min=1e-9))
        phi = cosine_m1 * self.cos_m2 - sine * self.sin_m2
        phi = torch.where(cosine_m1 > self.threshold, phi, cosine_m1)
        
        # Apply m3 (CosFace additive margin)
        phi = phi - self.m3
        
        # One-hot and combine
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        return output * self.scale


if __name__ == "__main__":
    # Quick test
    print("Testing Loss Functions...")
    
    batch_size = 32
    in_features = 512
    out_features = 1000
    
    # Create random embeddings and labels
    embeddings = torch.randn(batch_size, in_features)
    embeddings = F.normalize(embeddings, dim=1)
    labels = torch.randint(0, out_features, (batch_size,))
    
    # Test ArcFace
    arcface = ArcFace(in_features, out_features, scale=64.0, margin=0.5)
    logits = arcface(embeddings, labels)
    loss = F.cross_entropy(logits, labels)
    print(f"ArcFace - Logits shape: {logits.shape}, Loss: {loss.item():.4f}")
    
    # Test CosFace
    cosface = CosFace(in_features, out_features, scale=64.0, margin=0.35)
    logits = cosface(embeddings, labels)
    loss = F.cross_entropy(logits, labels)
    print(f"CosFace - Logits shape: {logits.shape}, Loss: {loss.item():.4f}")
    
    # Test Focal Loss
    focal = FocalLoss(gamma=2.0)
    focal_loss = focal(logits, labels)
    print(f"Focal Loss: {focal_loss.item():.4f}")
    
    # Test Combined Loss
    combined = CombinedLoss(in_features, out_features)
    combined_loss = combined(embeddings, labels)
    print(f"Combined Loss: {combined_loss.item():.4f}")
    
    print("\n✓ All loss tests passed!")
