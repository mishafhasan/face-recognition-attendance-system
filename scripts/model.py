"""
MobileFaceNet + ArcFace Model Architecture

Shared model definitions used by training, evaluation, and export scripts.
Architecture: MobileFaceNet with Bottleneck Residual Blocks + ArcFace Loss.

Fixes applied:
  - Upgraded DepthWise blocks to Bottleneck-style with channel expansion factor
    (previously plain DW-separable; expansion is required for sufficient model capacity)
  - Linear projection in Bottleneck (no activation after PW conv as per original paper)
  - Proper residual connection with skip only when stride==1 and channels match

MobileFaceNet: ~1.0M params, 112x112 input -> 512-D L2-normalized embedding
ArcFace: Additive Angular Margin Loss (scale=64, margin=0.5)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Building Blocks
# =============================================================================

class ConvBlock(nn.Module):
    """Conv2D -> BatchNorm -> PReLU"""

    def __init__(self, in_c, out_c, k=3, s=1, p=1, g=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthWise(nn.Module):
    """
    Depthwise Separable Convolution with Bottleneck Expansion.

    Structure (MobileNetV2-style inverted residual for face recognition):
        PW 1x1 (in_c -> in_c * t)  : expansion (PReLU)
        DW 3x3/stride               : spatial filtering (PReLU)
        PW 1x1 (in_c * t -> out_c) : projection (linear, NO activation)

    The linear projection at the end preserves the learned features in the
    embedding space without non-linear distortion — critical for face recognition.
    Expansion factor t=2 doubles internal channels, giving the model sufficient
    capacity to learn discriminative face features even with lightweight DW convs.
    """

    def __init__(self, in_c, out_c, s=1, t=2):
        super().__init__()
        mid_c = in_c * t

        # Channel expansion (pointwise)
        self.expand = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.PReLU(mid_c),
        )
        # Depthwise spatial conv
        self.dw = nn.Sequential(
            nn.Conv2d(mid_c, mid_c, 3, s, 1, groups=mid_c, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.PReLU(mid_c),
        )
        # Linear projection (no activation - preserves feature manifold)
        self.project = nn.Sequential(
            nn.Conv2d(mid_c, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x):
        x = self.expand(x)
        x = self.dw(x)
        return self.project(x)


class DepthWiseRes(nn.Module):
    """Bottleneck Depthwise block with optional residual connection."""

    def __init__(self, in_c, out_c, s=1, t=2):
        super().__init__()
        # Residual only when spatial dims and channels are preserved
        self.use_residual = (s == 1 and in_c == out_c)
        self.block = DepthWise(in_c, out_c, s, t)

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_residual else out


def make_stage(in_c, out_c, n, s=2, t=2):
    """Build a stage of n bottleneck depthwise blocks."""
    layers = [DepthWiseRes(in_c, out_c, s, t)]
    for _ in range(1, n):
        layers.append(DepthWiseRes(out_c, out_c, 1, t))
    return nn.Sequential(*layers)


# =============================================================================
# MobileFaceNet Backbone
# =============================================================================

class MobileFaceNet(nn.Module):
    """
    MobileFaceNet: Lightweight Face Recognition Backbone with Bottleneck Blocks

    Architecture:
        Conv3x3/2  (3 -> 64)                    : 112x112 -> 56x56
        DepthWise  (64 -> 64, stride 1)          : 56x56
        Stage 1:   5 bottleneck blocks (64->64, stride 2)     -> 28x28
        Stage 2:   1 bottleneck block  (64->128, stride 2)    -> 14x14
        Stage 3:   6 bottleneck blocks (128->128, stride 2)   -> 7x7
        Stage 4:   1 bottleneck block  (128->128, stride 1)   -> 7x7
        Conv1x1    (128 -> 512)
        GDC 7x7    (depthwise global)                         -> 1x1
        Linear 512 -> emb_dim
        BatchNorm1d -> L2-normalize

    Total: ~1.0M parameters  (bottleneck t=2)
    Output: 512-D L2-normalized embedding
    """

    def __init__(self, emb_dim=512):
        super().__init__()
        self.conv1 = ConvBlock(3, 64, 3, 2, 1)
        self.dw1 = DepthWise(64, 64, 1, t=1)   # t=1 for the initial DW (no expansion needed)
        self.stage1 = make_stage(64, 64, 5, 2, t=2)
        self.stage2 = make_stage(64, 128, 1, 2, t=2)
        self.stage3 = make_stage(128, 128, 6, 2, t=2)
        self.stage4 = make_stage(128, 128, 1, 1, t=2)
        self.conv_exp = ConvBlock(128, 512, 1, 1, 0)
        self.gdc = nn.Sequential(
            nn.Conv2d(512, 512, 7, 1, 0, groups=512, bias=False),
            nn.BatchNorm2d(512),
        )
        self.fc = nn.Linear(512, emb_dim, bias=False)
        self.bn = nn.BatchNorm1d(emb_dim)
        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv_exp(x)
        x = self.gdc(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        return F.normalize(x, p=2, dim=1)


# =============================================================================
# ArcFace Loss
# =============================================================================

class ArcFace(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss

    Features:
        - Numerically stable arccos-based margin (matches insightface official)
        - Margin warmup support via set_margin()
        - Stores raw cosine for accurate training accuracy measurement
        - FP32 forced internally for AMP stability

    Args:
        in_features: Embedding dimension (512)
        out_features: Number of classes/identities
        s: Scaling factor (default 64.0)
        m: Angular margin in radians (default 0.5, ~28.6 degrees)
    """

    def __init__(self, in_features, out_features, s=64.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self._last_cos = None  # Raw cosine for accuracy computation

    def set_margin(self, m):
        """Dynamically update margin (for margin warmup)."""
        self.m = m

    def forward(self, emb, labels):
        """
        Args:
            emb: [batch, 512] L2-normalized embeddings
            labels: [batch] class labels
        Returns:
            Scaled logits with angular margin applied
        """
        # Force FP32 for numerical stability under AMP
        emb_fp32 = emb.float()
        w = F.normalize(self.weight.float(), p=2, dim=1)
        cos = F.linear(emb_fp32, w).clamp(-1 + 1e-7, 1 - 1e-7)

        # Store raw cosine for accuracy measurement (before margin)
        self._last_cos = cos.detach()

        # Numerically stable: arccos -> add margin -> cos
        target_logit = cos[torch.arange(cos.size(0), device=cos.device), labels]
        target_theta = target_logit.arccos()
        target_theta_m = (target_theta + self.m).clamp(0, math.pi)
        cos_theta_m = target_theta_m.cos()

        # Replace correct-class cosine with margin-penalized version
        one_hot = F.one_hot(labels, self.weight.size(0)).float()
        logits = cos * (1 - one_hot) + cos_theta_m.unsqueeze(1) * one_hot

        return logits * self.s


# =============================================================================
# Combined Model
# =============================================================================

class MobileArcFace(nn.Module):
    """
    Complete Face Recognition Model: MobileFaceNet backbone + ArcFace head.

    During training: forward(images, labels) -> logits for cross-entropy
    During inference: forward(images) -> 512-D embeddings
    """

    def __init__(self, num_classes, emb_dim=512, s=64.0, m=0.5):
        super().__init__()
        self.backbone = MobileFaceNet(emb_dim)
        self.head = ArcFace(emb_dim, num_classes, s, m)

    def forward(self, x, labels=None):
        emb = self.backbone(x)
        if labels is not None:
            return self.head(emb, labels)
        return emb


if __name__ == "__main__":
    # Quick architecture verification
    model = MobileFaceNet(emb_dim=512)
    total_params = sum(p.numel() for p in model.parameters())

    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 112, 112)
        emb = model(x)

    print(f"MobileFaceNet")
    print(f"  Parameters: {total_params:,}")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {emb.shape}")
    print(f"  Norm:   {emb.norm().item():.4f}")
