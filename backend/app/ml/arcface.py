"""
ArcFace Loss Function for Face Recognition

Implements the Additive Angular Margin Loss (ArcFace) from the paper:
"ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
https://arxiv.org/abs/1801.07698

Key Features:
- Additive angular margin in the angular/geodesic space
- Better geometric interpretation than softmax
- SOTA face recognition performance
- Partial FC support for memory-efficient training with large class numbers

Mathematical Formulation:
L = -log(exp(s * cos(θ_y + m)) / (exp(s * cos(θ_y + m)) + Σ exp(s * cos(θ_j))))

Where:
- s: Scale factor (default: 64)
- m: Angular margin (default: 0.5 radians ≈ 28.6°)
- θ_y: Angle between embedding and target class weight
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import torch.distributed as dist


class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition.
    
    This loss function adds an angular margin to the target class angle,
    making the decision boundary in angular space more discriminative.
    
    Args:
        embedding_size: Size of face embeddings (default: 512)
        num_classes: Number of identity classes in training set
        scale: Scaling factor for logits (default: 64.0)
        margin: Angular margin in radians (default: 0.5 ≈ 28.6°)
        easy_margin: Use easy margin formulation (default: False)
        
    Example:
        >>> arcface = ArcFaceLoss(512, num_classes=10000, scale=64.0, margin=0.5)
        >>> embeddings = torch.randn(32, 512)  # Batch of normalized embeddings
        >>> labels = torch.randint(0, 10000, (32,))
        >>> loss = arcface(embeddings, labels)
    """
    
    def __init__(
        self,
        embedding_size: int = 512,
        num_classes: int = 10000,
        scale: float = 64.0,
        margin: float = 0.5,
        easy_margin: bool = False
    ):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        
        # Learnable class weight matrix W ∈ R^(num_classes × embedding_size)
        # Each row represents the prototype for one identity
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
        # Pre-compute margin values for efficiency
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)  # Threshold for numerical stability
        self.mm = math.sin(math.pi - margin) * margin
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ArcFace loss.
        
        Args:
            embeddings: L2-normalized face embeddings [B, embedding_size]
            labels: Ground truth class labels [B]
            
        Returns:
            Scalar loss value
        """
        # Normalize embeddings and weights (both to unit length)
        # This makes dot product equivalent to cosine similarity
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weights_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity: cos(θ) = W · x
        # Shape: [B, num_classes]
        cosine = F.linear(embeddings_norm, weights_norm)
        
        # Clamp for numerical stability
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        
        # Compute sin(θ) from cos(θ) using identity: sin²θ + cos²θ = 1
        sine = torch.sqrt(1.0 - cosine.pow(2))
        
        # Compute cos(θ + m) using angle addition formula:
        # cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            # Easy margin: only apply margin when cos(θ) > 0
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Standard ArcFace: apply threshold for numerical stability
            # When θ + m > π, use the linear approximation
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Create one-hot encoding for target classes
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # Apply margin only to the target class
        # output = cos(θ + m) for target class, cos(θ) for non-target classes
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # Scale the output
        output = output * self.scale
        
        # Compute cross-entropy loss
        loss = self.criterion(output, labels)
        
        return loss
    
    def get_logits(
        self, 
        embeddings: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get logits without computing loss (for inference or analysis).
        
        Args:
            embeddings: Face embeddings [B, embedding_size]
            labels: Optional labels (if provided, applies margin)
            
        Returns:
            Logits of shape [B, num_classes]
        """
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weights_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings_norm, weights_norm)
        
        if labels is not None:
            # Apply margin to target class
            cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
            sine = torch.sqrt(1.0 - cosine.pow(2))
            phi = cosine * self.cos_m - sine * self.sin_m
            
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, labels.view(-1, 1), 1)
            
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            return output * self.scale
        
        return cosine * self.scale


class CosFaceLoss(nn.Module):
    """
    CosFace (AM-Softmax): Additive Margin Softmax Loss.
    
    An alternative to ArcFace that adds margin directly to cosine similarity
    rather than to the angle. Simpler but slightly less effective than ArcFace.
    
    L = -log(exp(s * (cos(θ_y) - m)) / (exp(s * (cos(θ_y) - m)) + Σ exp(s * cos(θ_j))))
    
    Args:
        embedding_size: Size of face embeddings
        num_classes: Number of identity classes
        scale: Scaling factor (default: 64.0)
        margin: Cosine margin (default: 0.35)
    """
    
    def __init__(
        self,
        embedding_size: int = 512,
        num_classes: int = 10000,
        scale: float = 64.0,
        margin: float = 0.35
    ):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute CosFace loss."""
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weights_norm = F.normalize(self.weight, p=2, dim=1)
        
        cosine = F.linear(embeddings_norm, weights_norm)
        
        # Subtract margin from target class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        output = cosine - one_hot * self.margin
        output = output * self.scale
        
        return self.criterion(output, labels)


class PartialFC(nn.Module):
    """
    Partial FC for memory-efficient training with large class numbers.
    
    When training with millions of identities, the full classification head
    becomes too large for GPU memory. Partial FC samples a subset of classes
    per batch, significantly reducing memory usage while maintaining accuracy.
    
    Key Features:
    - Samples only a fraction of negative classes per batch
    - Always includes the positive (target) classes
    - Reduces memory from O(num_classes) to O(sample_ratio * num_classes)
    - Minimal impact on accuracy (~0.1-0.2% drop)
    
    Args:
        embedding_size: Size of face embeddings
        num_classes: Total number of identity classes
        sample_ratio: Fraction of classes to sample (default: 0.1 = 10%)
        scale: Scaling factor for logits
        margin: Angular margin for ArcFace
        
    Reference: https://arxiv.org/abs/2010.05222
    """
    
    def __init__(
        self,
        embedding_size: int = 512,
        num_classes: int = 1000000,
        sample_ratio: float = 0.1,
        scale: float = 64.0,
        margin: float = 0.5
    ):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.sample_ratio = sample_ratio
        self.scale = scale
        self.margin = margin
        
        # Full weight matrix (stored on CPU to save GPU memory)
        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, embedding_size),
            requires_grad=True
        )
        nn.init.xavier_uniform_(self.weight)
        
        # Pre-compute margin values
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        
        # Number of classes to sample per batch
        self.num_sample = int(num_classes * sample_ratio)
        
    def forward(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Partial FC loss.
        
        Args:
            embeddings: L2-normalized face embeddings [B, embedding_size]
            labels: Ground truth class labels [B]
            
        Returns:
            Tuple of (loss, accuracy)
        """
        device = embeddings.device
        batch_size = embeddings.size(0)
        
        # Get unique labels in batch
        unique_labels = labels.unique()
        
        # Sample additional random classes (excluding those in batch)
        all_indices = torch.arange(self.num_classes, device=device)
        
        # Create mask for batch labels
        mask = torch.ones(self.num_classes, dtype=torch.bool, device=device)
        mask[unique_labels] = False
        
        # Sample from remaining classes
        remaining_indices = all_indices[mask]
        num_to_sample = min(self.num_sample, len(remaining_indices))
        
        if num_to_sample > 0:
            sample_perm = torch.randperm(len(remaining_indices), device=device)
            sampled_indices = remaining_indices[sample_perm[:num_to_sample]]
            
            # Combine batch labels with sampled classes
            selected_indices = torch.cat([unique_labels, sampled_indices])
        else:
            selected_indices = unique_labels
        
        # Map original labels to new indices
        label_map = {int(old): new for new, old in enumerate(unique_labels.tolist())}
        mapped_labels = torch.tensor(
            [label_map[int(l)] for l in labels], 
            device=device, 
            dtype=torch.long
        )
        
        # Get selected weights (move to GPU)
        selected_weights = self.weight[selected_indices].to(device)
        
        # Compute cosine similarity
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weights_norm = F.normalize(selected_weights, p=2, dim=1)
        cosine = F.linear(embeddings_norm, weights_norm)
        
        # Apply ArcFace margin
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt(1.0 - cosine.pow(2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Create one-hot for mapped labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, mapped_labels.view(-1, 1), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale
        
        # Compute loss and accuracy
        loss = F.cross_entropy(output, mapped_labels)
        
        with torch.no_grad():
            pred = output.argmax(dim=1)
            accuracy = (pred == mapped_labels).float().mean()
        
        return loss, accuracy


class CombinedMarginLoss(nn.Module):
    """
    Combined Margin Loss (ArcFace + CosFace + SphereFace).
    
    Allows combining different margin types for flexible decision boundaries.
    
    L = -log(exp(s * (cos(m1 * θ + m2) - m3)) / ...)
    
    Args:
        embedding_size: Size of face embeddings
        num_classes: Number of identity classes
        scale: Scaling factor
        m1: SphereFace-style multiplicative margin (default: 1.0)
        m2: ArcFace-style additive angular margin (default: 0.5)
        m3: CosFace-style additive cosine margin (default: 0.0)
    """
    
    def __init__(
        self,
        embedding_size: int = 512,
        num_classes: int = 10000,
        scale: float = 64.0,
        m1: float = 1.0,
        m2: float = 0.5,
        m3: float = 0.0
    ):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.scale = scale
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined margin loss."""
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weights_norm = F.normalize(self.weight, p=2, dim=1)
        
        cosine = F.linear(embeddings_norm, weights_norm)
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        
        # Apply combined margin: cos(m1 * θ + m2) - m3
        theta = torch.acos(cosine)
        target_theta = self.m1 * theta + self.m2
        target_cosine = torch.cos(target_theta) - self.m3
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        output = (one_hot * target_cosine) + ((1.0 - one_hot) * cosine)
        output = output * self.scale
        
        return self.criterion(output, labels)


def create_arcface_loss(
    embedding_size: int = 512,
    num_classes: int = 10000,
    scale: float = 64.0,
    margin: float = 0.5,
    loss_type: str = "arcface",
    **kwargs
) -> nn.Module:
    """
    Factory function to create face recognition loss.
    
    Args:
        embedding_size: Embedding dimension
        num_classes: Number of identity classes
        scale: Scaling factor
        margin: Margin value
        loss_type: 'arcface', 'cosface', 'combined', or 'partial_fc'
        
    Returns:
        Loss module
    """
    if loss_type == "arcface":
        return ArcFaceLoss(embedding_size, num_classes, scale, margin, **kwargs)
    elif loss_type == "cosface":
        return CosFaceLoss(embedding_size, num_classes, scale, margin)
    elif loss_type == "combined":
        return CombinedMarginLoss(embedding_size, num_classes, scale, **kwargs)
    elif loss_type == "partial_fc":
        return PartialFC(embedding_size, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    print("=" * 60)
    print("ArcFace Loss Function Test")
    print("=" * 60)
    
    # Test parameters
    embedding_size = 512
    num_classes = 10000
    batch_size = 32
    
    # Create loss functions
    arcface = ArcFaceLoss(embedding_size, num_classes, scale=64.0, margin=0.5)
    cosface = CosFaceLoss(embedding_size, num_classes, scale=64.0, margin=0.35)
    combined = CombinedMarginLoss(embedding_size, num_classes, scale=64.0, 
                                   m1=1.0, m2=0.5, m3=0.0)
    
    # Generate test data
    embeddings = torch.randn(batch_size, embedding_size)
    embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalize
    labels = torch.randint(0, num_classes, (batch_size,))
    
    print(f"\n📊 Test Configuration:")
    print(f"   Embedding size: {embedding_size}")
    print(f"   Num classes: {num_classes}")
    print(f"   Batch size: {batch_size}")
    
    print(f"\n🧪 Testing Loss Functions:")
    
    # Test ArcFace
    loss_arc = arcface(embeddings, labels)
    print(f"   ArcFace Loss: {loss_arc.item():.4f}")
    
    # Test CosFace
    loss_cos = cosface(embeddings, labels)
    print(f"   CosFace Loss: {loss_cos.item():.4f}")
    
    # Test Combined
    loss_combined = combined(embeddings, labels)
    print(f"   Combined Loss: {loss_combined.item():.4f}")
    
    # Test gradient flow
    print(f"\n🔄 Testing Gradient Flow:")
    
    embeddings.requires_grad = True
    loss = arcface(embeddings, labels)
    loss.backward()
    
    print(f"   Embeddings grad shape: {embeddings.grad.shape}")
    print(f"   Embeddings grad mean: {embeddings.grad.mean().item():.6f}")
    print(f"   Weight grad mean: {arcface.weight.grad.mean().item():.6f}")
    
    # Test Partial FC with smaller class count
    print(f"\n🧪 Testing Partial FC:")
    partial_fc = PartialFC(embedding_size, num_classes=1000, sample_ratio=0.1)
    
    embeddings_pfc = torch.randn(batch_size, embedding_size)
    embeddings_pfc = F.normalize(embeddings_pfc, p=2, dim=1)
    labels_pfc = torch.randint(0, 1000, (batch_size,))
    
    loss_pfc, acc_pfc = partial_fc(embeddings_pfc, labels_pfc)
    print(f"   Partial FC Loss: {loss_pfc.item():.4f}")
    print(f"   Partial FC Accuracy: {acc_pfc.item():.4f}")
    
    # Memory comparison
    print(f"\n💾 Memory Comparison:")
    full_params = num_classes * embedding_size * 4 / (1024 * 1024)
    partial_params = int(num_classes * 0.1) * embedding_size * 4 / (1024 * 1024)
    print(f"   Full FC memory: {full_params:.2f} MB")
    print(f"   Partial FC (10%): {partial_params:.2f} MB")
    print(f"   Memory savings: {(1 - partial_params/full_params)*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("✅ ArcFace Loss test completed successfully!")
    print("=" * 60)
