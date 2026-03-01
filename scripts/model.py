"""
MobileFaceNet + ArcFace Model Architecture

Shared model definitions used by training, evaluation, and export scripts.
Architecture matches the complete pipeline notebook exactly.

MobileFaceNet: ~1M params, 112x112 input -> 512-D L2-normalized embedding
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
    """Depthwise Separable Convolution: DW 3x3 -> PW 1x1"""

    def __init__(self, in_c, out_c, s=1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, 3, s, 1, groups=in_c, bias=False)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.act1 = nn.PReLU(in_c)
        self.pw = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act2 = nn.PReLU(out_c)

    def forward(self, x):
        x = self.act1(self.bn1(self.dw(x)))
        return self.act2(self.bn2(self.pw(x)))


class DepthWiseRes(nn.Module):
    """Depthwise Separable Conv with optional residual connection."""

    def __init__(self, in_c, out_c, s=1):
        super().__init__()
        self.residual = (s == 1 and in_c == out_c)
        self.dw = DepthWise(in_c, out_c, s)

    def forward(self, x):
        return x + self.dw(x) if self.residual else self.dw(x)


def make_stage(in_c, out_c, n, s=2):
    """Build a stage of n depthwise residual blocks."""
    layers = [DepthWiseRes(in_c, out_c, s)]
    for _ in range(1, n):
        layers.append(DepthWiseRes(out_c, out_c, 1))
    return nn.Sequential(*layers)


# =============================================================================
# MobileFaceNet Backbone
# =============================================================================

class MobileFaceNet(nn.Module):
    """
    MobileFaceNet: Lightweight Face Recognition Backbone

    Architecture:
        Conv3x3/2 (3->64) -> DepthWise (64->64)
        Stage 1: 5 blocks (64->64, stride 2)   112->56->28
        Stage 2: 1 block  (64->128, stride 2)  28->14
        Stage 3: 6 blocks (128->128, stride 2)  14->7
        Stage 4: 1 block  (128->128, stride 1)  7->7
        Conv1x1 (128->512) -> GDC 7x7 -> Linear -> BN -> L2-norm

    Total: ~1M parameters
    Output: 512-D L2-normalized embedding
    """

    def __init__(self, emb_dim=512):
        super().__init__()
        self.conv1 = ConvBlock(3, 64, 3, 2, 1)
        self.dw1 = DepthWise(64, 64, 1)
        self.stage1 = make_stage(64, 64, 5, 2)
        self.stage2 = make_stage(64, 128, 1, 2)
        self.stage3 = make_stage(128, 128, 6, 2)
        self.stage4 = make_stage(128, 128, 1, 1)
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
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

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
