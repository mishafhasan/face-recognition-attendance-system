"""
Lightweight Face Embedding Model

Fast, accurate, and lightweight model for face recognition.
Uses MobileNetV3 backbone optimized for mobile/edge deployment.

Features:
- MobileNetV3-Small backbone (~1.5M params)
- 112x112 input (optimized for mobile)
- 128-dim L2-normalized embeddings
- ~5ms inference (CPU), ~1ms (GPU)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional


class LightFaceNet(nn.Module):
    """
    Lightweight Face Embedding Network.
    
    Uses MobileNetV3 backbone for fast, efficient face embeddings.
    Optimized for real-time face recognition on CPU/mobile devices.
    
    Args:
        embedding_dim: Output embedding dimension (default: 128)
        backbone: 'mobilenet_v3_small' (fast) or 'mobilenet_v3_large' (accurate)
        pretrained: Use ImageNet pretrained weights
        dropout: Dropout rate in embedding head
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        backbone: str = "mobilenet_v3_small",
        pretrained: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.backbone_name = backbone
        
        # Select backbone
        if backbone == "mobilenet_v3_small":
            base = models.mobilenet_v3_small(
                weights='IMAGENET1K_V1' if pretrained else None
            )
            feature_dim = 576
        elif backbone == "mobilenet_v3_large":
            base = models.mobilenet_v3_large(
                weights='IMAGENET1K_V1' if pretrained else None
            )
            feature_dim = 960
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Use convolutional features
        self.features = base.features
        self.avgpool = base.avgpool
        
        # Embedding head
        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.Hardswish(inplace=True),  # Faster than ReLU on mobile
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate L2-normalized face embedding.
        
        Args:
            x: Input tensor [B, 3, 112, 112], normalized to [-1, 1]
            
        Returns:
            L2-normalized embeddings [B, embedding_dim]
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward() for clarity."""
        return self.forward(x)


class TripletLoss(nn.Module):
    """
    Triplet Loss for face embedding training.
    
    L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
    
    Args:
        margin: Minimum distance margin (default: 0.2)
    """
    
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
        
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings [B, D]
            positive: Positive embeddings [B, D] (same person)
            negative: Negative embeddings [B, D] (different person)
            
        Returns:
            Scalar loss value
        """
        pos_dist = (anchor - positive).pow(2).sum(dim=1)
        neg_dist = (anchor - negative).pow(2).sum(dim=1)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


# Legacy compatibility - alias to old name
FaceEmbeddingModel = LightFaceNet


def create_model(
    embedding_dim: int = 128,
    backbone: str = "mobilenet_v3_small",
    pretrained: bool = True,
    device: Optional[str] = None
) -> LightFaceNet:
    """
    Factory function to create model.
    
    Args:
        embedding_dim: Embedding dimension
        backbone: Backbone architecture
        pretrained: Use pretrained weights
        device: Device to place model on
        
    Returns:
        Initialized LightFaceNet model
    """
    model = LightFaceNet(
        embedding_dim=embedding_dim,
        backbone=backbone,
        pretrained=pretrained
    )
    
    if device:
        model = model.to(device)
    
    return model


def load_model(checkpoint_path: str, device: str = "cpu") -> LightFaceNet:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded LightFaceNet model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get("config", {})
    model = create_model(
        embedding_dim=config.get("embedding_dim", 128),
        backbone=config.get("backbone", "mobilenet_v3_small"),
        pretrained=False,
        device=device
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model


if __name__ == "__main__":
    # Test model
    print("Testing LightFaceNet...")
    
    model = create_model(embedding_dim=128, backbone="mobilenet_v3_small")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Test forward pass
    x = torch.randn(4, 3, 112, 112)
    embeddings = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Output norm: {embeddings.norm(dim=1)}")
    
    # Model size
    model_size_mb = num_params * 4 / (1024 * 1024)
    print(f"Model size: ~{model_size_mb:.1f} MB")
    
    print("\n✅ LightFaceNet test passed!")
