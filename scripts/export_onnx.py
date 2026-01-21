#!/usr/bin/env python3
"""
Export MobileFaceNet model to ONNX format.

Usage:
    python export_onnx.py --checkpoint path/to/checkpoint.pth --output model.onnx
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend' / 'app'))


def load_model(checkpoint_path: str = None, embedding_dim: int = 512):
    """Load MobileFaceNet model."""
    try:
        from ml.mobilefacenet import MobileFaceNet
        model = MobileFaceNet(embedding_dim=embedding_dim, use_eca=True)
    except ImportError:
        # Fallback: inline definition
        import math
        
        class ECABlock(nn.Module):
            def __init__(self, channels, gamma=2, b=1):
                super().__init__()
                k = int(abs((math.log2(channels) + b) / gamma))
                k = k if k % 2 else k + 1
                k = max(3, k)
                self.avg_pool = nn.AdaptiveAvgPool2d(1)
                self.conv = nn.Conv1d(1, 1, k, padding=k//2, bias=False)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
                y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
                return x * self.sigmoid(y)
        
        class InvertedResidual(nn.Module):
            def __init__(self, in_ch, out_ch, stride, expand_ratio, use_eca=True):
                super().__init__()
                hidden = in_ch * expand_ratio
                self.use_res = stride == 1 and in_ch == out_ch
                layers = []
                if expand_ratio != 1:
                    layers += [nn.Conv2d(in_ch, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.PReLU(hidden)]
                layers += [nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False), nn.BatchNorm2d(hidden), nn.PReLU(hidden)]
                if use_eca:
                    layers.append(ECABlock(hidden))
                layers += [nn.Conv2d(hidden, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)]
                self.conv = nn.Sequential(*layers)
            
            def forward(self, x):
                return x + self.conv(x) if self.use_res else self.conv(x)
        
        class MobileFaceNet(nn.Module):
            def __init__(self, embedding_dim=512, use_eca=True):
                super().__init__()
                self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU(64))
                self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, groups=64, bias=False), nn.BatchNorm2d(64), nn.PReLU(64))
                settings = [(2,64,5,2), (4,128,1,2), (2,128,6,1), (4,128,1,2), (2,128,2,1)]
                layers, in_ch = [], 64
                for exp, out, n, s in settings:
                    for i in range(n):
                        layers.append(InvertedResidual(in_ch, out, s if i==0 else 1, exp, use_eca))
                        in_ch = out
                self.bottlenecks = nn.Sequential(*layers)
                self.conv3 = nn.Sequential(nn.Conv2d(128, 512, 1, bias=False), nn.BatchNorm2d(512), nn.PReLU(512))
                self.conv4 = nn.Sequential(nn.Conv2d(512, 512, 7, groups=512, bias=False), nn.BatchNorm2d(512))
                self.fc = nn.Linear(512, embedding_dim, bias=False)
                self.bn = nn.BatchNorm1d(embedding_dim)
            
            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.bottlenecks(x)
                x = self.conv3(x)
                x = self.conv4(x)
                x = self.fc(x.flatten(1))
                x = self.bn(x)
                return F.normalize(x, p=2, dim=1)
        
        model = MobileFaceNet(embedding_dim=embedding_dim, use_eca=True)
    
    # Load weights if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print("Using random weights (no checkpoint provided)")
    
    model.eval()
    return model


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_size: tuple = (1, 3, 112, 112),
    opset_version: int = 14,
    dynamic_batch: bool = True
):
    """Export model to ONNX format."""
    model.eval()
    
    # Create example input
    example_input = torch.randn(input_size)
    
    # Dynamic axes for variable batch size
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'embedding': {0: 'batch_size'}
        }
    
    # Export
    torch.onnx.export(
        model,
        example_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['embedding'],
        dynamic_axes=dynamic_axes
    )
    
    print(f"ONNX model exported to {output_path}")
    
    # Validate
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation passed")
    except ImportError:
        print("ONNX not installed, skipping validation")
    except Exception as e:
        print(f"ONNX validation failed: {e}")
    
    # Test with ONNX Runtime
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_path)
        
        # Compare outputs
        ort_input = example_input.numpy()
        ort_output = session.run(None, {'input': ort_input})[0]
        
        with torch.no_grad():
            torch_output = model(example_input).numpy()
        
        diff = np.abs(torch_output - ort_output).max()
        print(f"Max output difference (PyTorch vs ONNX): {diff:.8f}")
        
        if diff < 1e-5:
            print("✓ Outputs match!")
        else:
            print("⚠ Small numerical differences detected")
    
    except ImportError:
        print("ONNX Runtime not installed, skipping verification")
    
    # Print file info
    file_size = Path(output_path).stat().st_size / 1024 / 1024
    print(f"File size: {file_size:.2f} MB")


def export_to_torchscript(
    model: nn.Module,
    output_path: str,
    input_size: tuple = (1, 3, 112, 112),
    optimize: bool = True
):
    """Export model to TorchScript format."""
    model.eval()
    
    example_input = torch.randn(input_size)
    
    # Trace
    traced_model = torch.jit.trace(model, example_input)
    
    # Optimize
    if optimize:
        traced_model = torch.jit.optimize_for_inference(traced_model)
    
    # Save
    traced_model.save(output_path)
    print(f"TorchScript model saved to {output_path}")
    
    # Verify
    loaded_model = torch.jit.load(output_path)
    
    with torch.no_grad():
        original_output = model(example_input)
        loaded_output = loaded_model(example_input)
        diff = (original_output - loaded_output).abs().max().item()
        print(f"Max output difference: {diff:.8f}")
    
    file_size = Path(output_path).stat().st_size / 1024 / 1024
    print(f"File size: {file_size:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Export MobileFaceNet to ONNX/TorchScript')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='mobilefacenet.onnx',
                        help='Output path')
    parser.add_argument('--format', type=str, choices=['onnx', 'torchscript', 'both'],
                        default='onnx', help='Export format')
    parser.add_argument('--embedding-dim', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--opset', type=int, default=14,
                        help='ONNX opset version')
    parser.add_argument('--no-dynamic-batch', action='store_true',
                        help='Disable dynamic batch size')
    args = parser.parse_args()
    
    print("=" * 50)
    print("MobileFaceNet Model Export")
    print("=" * 50)
    
    # Load model
    model = load_model(args.checkpoint, args.embedding_dim)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export
    if args.format in ['onnx', 'both']:
        onnx_path = str(output_path) if args.format == 'onnx' else str(output_path.with_suffix('.onnx'))
        print(f"\n--- Exporting to ONNX ---")
        export_to_onnx(
            model,
            onnx_path,
            opset_version=args.opset,
            dynamic_batch=not args.no_dynamic_batch
        )
    
    if args.format in ['torchscript', 'both']:
        ts_path = str(output_path) if args.format == 'torchscript' else str(output_path.with_suffix('.pt'))
        print(f"\n--- Exporting to TorchScript ---")
        export_to_torchscript(model, ts_path)
    
    print("\n" + "=" * 50)
    print("Export Complete!")
    print("=" * 50)


if __name__ == '__main__':
    main()
