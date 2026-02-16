"""MobileFaceNet Architecture for Face Recognition.

This module implements the MobileFaceNet architecture, a lightweight backbone
specifically designed for face recognition on mobile and embedded devices.

Architecture Features:
    - Depthwise Separable Convolutions for efficiency
    - PReLU activations (learnable slope)
    - Global Depthwise Conv instead of GAP
    - ~1M parameters (25x smaller than ResNet-50)
    - 128/512-dimensional normalized embeddings

Reference:
    Chen et al., "MobileFaceNets: Efficient CNNs for Accurate Real-time 
    Face Verification on Mobile Devices", CCBR 2018
    https://arxiv.org/abs/1804.07573
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Standard Convolution Block: Conv2d -> BatchNorm -> PReLU.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Convolution padding
        groups: Number of groups for grouped convolution
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.prelu(self.bn(self.conv(x)))


class DepthWise(nn.Module):
    """Depthwise Separable Convolution Block.
    
    Consists of:
        1. Depthwise Conv: Spatial filtering per channel
        2. Pointwise Conv: Channel mixing (1x1 conv)
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Depthwise kernel size
        stride: Depthwise stride
        padding: Depthwise padding
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        # Depthwise convolution (spatial)
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.prelu1 = nn.PReLU(in_channels)
        
        # Pointwise convolution (channel mixing)
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.prelu2 = nn.PReLU(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.prelu1(self.bn1(self.depthwise(x)))
        x = self.prelu2(self.bn2(self.pointwise(x)))
        return x


class DepthWiseResidual(nn.Module):
    """Depthwise Separable Convolution with Residual Connection.
    
    Similar to MobileNetV2 inverted residual but with PReLU.
    Residual connection helps gradient flow for deeper networks.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Depthwise kernel size
        stride: Depthwise stride
        padding: Depthwise padding
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.use_residual = (in_channels == out_channels) and (stride == 1)
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.prelu1 = nn.PReLU(in_channels)
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.prelu2 = nn.PReLU(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional residual connection."""
        identity = x
        
        out = self.prelu1(self.bn1(self.depthwise(x)))
        out = self.prelu2(self.bn2(self.pointwise(out)))
        
        if self.use_residual:
            out = out + identity
        
        return out


def make_stage(
    in_channels: int,
    out_channels: int,
    num_blocks: int,
    stride: int = 2,
) -> nn.Sequential:
    """Create a stage of depthwise residual blocks.
    
    First block may downsample (stride > 1), rest maintain spatial size.
    
    Args:
        in_channels: Input channels for first block
        out_channels: Output channels for all blocks
        num_blocks: Number of blocks in stage
        stride: Stride for first block (downsampling)
    
    Returns:
        Sequential container with all blocks
    """
    layers = []
    
    # First block with potential downsampling
    layers.append(DepthWiseResidual(in_channels, out_channels, stride=stride))
    
    # Remaining blocks maintain spatial size
    for _ in range(1, num_blocks):
        layers.append(DepthWiseResidual(out_channels, out_channels, stride=1))
    
    return nn.Sequential(*layers)


class MobileFaceNet(nn.Module):
    """MobileFaceNet Backbone for Face Recognition.
    
    A lightweight CNN backbone (~1M parameters) specifically designed
    for face recognition. Uses depthwise separable convolutions and
    global depthwise convolution for efficient feature extraction.
    
    Architecture:
        - Initial Conv: 3 -> 64 channels
        - Stage 1: 64 channels, 5 residual blocks
        - Stage 2: 64 -> 128 channels, 6 residual blocks  
        - Stage 3: 128 channels, 2 residual blocks
        - Global Depthwise Conv 7x7
        - Embedding projection (1x1 conv)
        - L2 normalization
    
    Args:
        embedding_size: Size of output embedding (default: 512)
        input_size: Input image size (default: 112)
        
    Input:
        x: (B, 3, 112, 112) RGB face images normalized to [-1, 1]
        
    Output:
        embeddings: (B, embedding_size) L2-normalized embeddings
    """
    
    def __init__(self, embedding_size: int = 512, input_size: int = 112):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.input_size = input_size
        
        # Calculate feature map size after stages
        # 112 -> 56 (stride 2) -> 28 (stride 2) -> 14 (stride 2) -> 7 (stride 2)
        self.feature_size = input_size // 16  # = 7 for 112x112 input
        
        # Initial convolution: 3 -> 64 channels, stride 2
        self.conv1 = ConvBlock(3, 64, kernel_size=3, stride=2, padding=1)
        
        # Depthwise block before stages
        self.dw_conv = DepthWise(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Stage 1: 64 channels, 5 blocks, stride 2 for first block
        self.stage1 = make_stage(64, 64, num_blocks=5, stride=2)
        
        # Stage 2: 64 -> 128 channels, 6 blocks, stride 2 for first block
        self.stage2 = make_stage(64, 128, num_blocks=6, stride=2)
        
        # Stage 3: 128 channels, 2 blocks, stride 2 for first block
        self.stage3 = make_stage(128, 128, num_blocks=2, stride=2)
        
        # Expand to 512 channels
        self.conv_expand = ConvBlock(128, 512, kernel_size=1, stride=1, padding=0)
        
        # Global Depthwise Convolution (instead of GAP)
        # This preserves more spatial information
        self.global_dw = nn.Conv2d(
            512, 512, 
            kernel_size=self.feature_size,  # 7x7 for 112 input
            stride=1, 
            padding=0, 
            groups=512, 
            bias=False
        )
        self.global_bn = nn.BatchNorm2d(512)
        
        # Embedding projection
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """Extract face embeddings.
        
        Args:
            x: Input tensor of shape (B, 3, 112, 112)
            return_features: If True, return features before normalization
            
        Returns:
            L2-normalized embeddings of shape (B, embedding_size)
        """
        # Initial convolution
        x = self.conv1(x)           # (B, 64, 56, 56)
        x = self.dw_conv(x)         # (B, 64, 56, 56)
        
        # Stages
        x = self.stage1(x)          # (B, 64, 28, 28)
        x = self.stage2(x)          # (B, 128, 14, 14)
        x = self.stage3(x)          # (B, 128, 7, 7)
        
        # Expand channels
        x = self.conv_expand(x)     # (B, 512, 7, 7)
        
        # Global depthwise + embedding
        x = self.global_bn(self.global_dw(x))  # (B, 512, 1, 1)
        features = self.embedding(x)            # (B, embedding_size)
        
        if return_features:
            return features
        
        # L2 normalize
        embeddings = F.normalize(features, p=2, dim=1)
        
        return embeddings
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward() for clarity in inference code."""
        return self.forward(x)
    
    def get_embedding_size(self) -> int:
        """Return the embedding size."""
        return self.embedding_size


class MobileArcFaceModel(nn.Module):
    """Complete MobileArcFace model for training.
    
    Combines MobileFaceNet backbone with ArcFace/CosFace classification head.
    Used during training to leverage the angular margin loss.
    For inference, use only the backbone's embeddings.
    
    Args:
        num_classes: Number of identity classes in training set
        embedding_size: Size of face embeddings (default: 512)
        loss_type: Type of margin loss ('arcface' or 'cosface')
        scale: Scaling factor for logits
        margin: Angular margin in radians
        
    Example:
        >>> model = MobileArcFaceModel(num_classes=10000)
        >>> images = torch.randn(32, 3, 112, 112)
        >>> labels = torch.randint(0, 10000, (32,))
        >>> logits = model(images, labels)  # Training
        >>> embeddings = model.get_embedding(images)  # Inference
    """
    
    def __init__(
        self,
        num_classes: int,
        embedding_size: int = 512,
        loss_type: str = "arcface",
        scale: float = 64.0,
        margin: float = 0.5,
        backbone: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = MobileFaceNet(embedding_size=embedding_size)
        
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.loss_type = loss_type
        
        # Import loss from losses module
        from .losses import ArcFace, CosFace
        
        if loss_type == "arcface":
            self.head = ArcFace(
                in_features=embedding_size,
                out_features=num_classes,
                scale=scale,
                margin=margin,
            )
        elif loss_type == "cosface":
            self.head = CosFace(
                in_features=embedding_size,
                out_features=num_classes,
                scale=scale,
                margin=margin,
            )
        else:
            # Simple linear classifier fallback
            self.head = nn.Linear(embedding_size, num_classes, bias=False)
    
    def forward(
        self, 
        x: torch.Tensor, 
        labels: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        """Forward pass for training.
        
        Args:
            x: Input images (B, 3, 112, 112)
            labels: Ground truth labels (B,) - required for margin loss
            return_embeddings: If True, return (logits, embeddings)
            
        Returns:
            logits: Class logits (B, num_classes) for cross-entropy loss
        """
        embeddings = self.backbone(x)
        
        if labels is not None:
            if self.loss_type in ["arcface", "cosface"]:
                logits = self.head(embeddings, labels)
            else:
                logits = self.head(embeddings)
            
            if return_embeddings:
                return logits, embeddings
            return logits
        else:
            # Return raw embeddings if no labels (inference mode)
            return embeddings
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get face embeddings for inference.
        
        Args:
            x: Input images (B, 3, 112, 112)
            
        Returns:
            embeddings: L2-normalized embeddings (B, embedding_size)
        """
        return self.backbone.get_embedding(x)
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for get_embedding."""
        return self.get_embedding(x)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_backbone_parameters(self) -> int:
        """Count backbone parameters only."""
        return sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)


def create_mobile_arcface(
    num_classes: int,
    embedding_size: int = 512,
    loss_type: str = "arcface",
    scale: float = 64.0,
    margin: float = 0.5,
    pretrained: Optional[str] = None,
) -> MobileArcFaceModel:
    """Factory function to create MobileArcFace model.
    
    Args:
        num_classes: Number of training identities
        embedding_size: Embedding dimension
        loss_type: 'arcface' or 'cosface'
        scale: Logit scaling factor
        margin: Angular margin
        pretrained: Path to pretrained weights (optional)
        
    Returns:
        MobileArcFaceModel instance
    """
    model = MobileArcFaceModel(
        num_classes=num_classes,
        embedding_size=embedding_size,
        loss_type=loss_type,
        scale=scale,
        margin=margin,
    )
    
    if pretrained is not None:
        state_dict = torch.load(pretrained, map_location='cpu')
        # Handle both full checkpoint and state_dict only
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {pretrained}")
    
    return model


def create_mobilefacenet(
    embedding_size: int = 512,
    pretrained: Optional[str] = None,
    device: str = 'cpu'
) -> MobileFaceNet:
    """Create MobileFaceNet backbone only (for inference).
    
    Args:
        embedding_size: Embedding dimension
        pretrained: Path to pretrained weights
        device: Device to load model on
        
    Returns:
        MobileFaceNet backbone
    """
    model = MobileFaceNet(embedding_size=embedding_size)
    
    if pretrained is not None:
        state_dict = torch.load(pretrained, map_location=device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        # Filter to only backbone weights
        backbone_state = {k.replace('backbone.', ''): v for k, v in state_dict.items() 
                         if k.startswith('backbone.')}
        if backbone_state:
            model.load_state_dict(backbone_state, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
    
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    print("Testing MobileFaceNet...")
    
    # Create model
    backbone = MobileFaceNet(embedding_size=512)
    num_params = sum(p.numel() for p in backbone.parameters())
    print(f"Backbone parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Test forward pass
    x = torch.randn(4, 3, 112, 112)
    embeddings = backbone(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Embedding norm: {torch.norm(embeddings, dim=1)}")  # Should be 1.0
    
    # Test full model
    print("\nTesting MobileArcFaceModel...")
    model = MobileArcFaceModel(num_classes=1000, embedding_size=512)
    total_params = model.count_parameters()
    print(f"Total parameters: {total_params:,}")
    
    labels = torch.randint(0, 1000, (4,))
    logits = model(x, labels)
    print(f"Logits shape: {logits.shape}")
    
    print("\n✓ All tests passed!")
