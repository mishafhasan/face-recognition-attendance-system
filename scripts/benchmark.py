#!/usr/bin/env python3
"""
Benchmark script for MobileFaceNet face recognition model.

Measures:
- Inference speed (CPU/GPU)
- Memory usage
- Embedding quality
"""

import argparse
import time
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try importing ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: nn.Module) -> float:
    """Get model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


def benchmark_pytorch(
    model: nn.Module,
    input_size: tuple = (1, 3, 112, 112),
    num_runs: int = 100,
    warmup: int = 10,
    device: str = 'cpu'
) -> dict:
    """Benchmark PyTorch model."""
    model = model.to(device)
    model.eval()
    
    x = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    
    times = np.array(times)
    
    return {
        'mean_ms': float(times.mean()),
        'std_ms': float(times.std()),
        'min_ms': float(times.min()),
        'max_ms': float(times.max()),
        'fps': float(1000 / times.mean())
    }


def benchmark_onnx(
    onnx_path: str,
    input_size: tuple = (1, 3, 112, 112),
    num_runs: int = 100,
    warmup: int = 10,
    use_gpu: bool = False
) -> dict:
    """Benchmark ONNX Runtime."""
    if not ONNX_AVAILABLE:
        print("ONNX Runtime not available")
        return {}
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    input_name = session.get_inputs()[0].name
    x = np.random.randn(*input_size).astype(np.float32)
    
    # Warmup
    for _ in range(warmup):
        _ = session.run(None, {input_name: x})
    
    # Measure
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = session.run(None, {input_name: x})
        times.append((time.perf_counter() - start) * 1000)
    
    times = np.array(times)
    
    return {
        'mean_ms': float(times.mean()),
        'std_ms': float(times.std()),
        'min_ms': float(times.min()),
        'max_ms': float(times.max()),
        'fps': float(1000 / times.mean()),
        'provider': session.get_providers()[0]
    }


def benchmark_memory(model: nn.Module, input_size: tuple = (1, 3, 112, 112)) -> dict:
    """Benchmark memory usage."""
    if not torch.cuda.is_available():
        return {'error': 'CUDA not available'}
    
    model = model.cuda()
    model.eval()
    
    torch.cuda.reset_peak_memory_stats()
    
    x = torch.randn(input_size).cuda()
    
    with torch.no_grad():
        _ = model(x)
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    # With AMP
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        _ = model(x)
    
    peak_memory_amp = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    return {
        'peak_memory_mb': peak_memory,
        'peak_memory_amp_mb': peak_memory_amp,
        'reduction_pct': (1 - peak_memory_amp / peak_memory) * 100
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark face recognition model')
    parser.add_argument('--model-path', type=str, default='../models/exported/mobilefacenet.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--num-runs', type=int, default=100, help='Number of benchmark runs')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup runs')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Face Recognition Model Benchmark")
    print("=" * 60)
    
    input_size = (args.batch_size, 3, 112, 112)
    print(f"\nInput size: {input_size}")
    print(f"Runs: {args.num_runs}")
    
    # Import model
    sys.path.insert(0, '../backend/app')
    try:
        from ml.mobilefacenet import MobileFaceNet
        model = MobileFaceNet(embedding_dim=512, use_eca=True)
        model.eval()
        
        print(f"\n--- Model Info ---")
        print(f"Parameters: {count_parameters(model):,}")
        print(f"Size: {get_model_size(model):.2f} MB")
        
        # PyTorch CPU
        print(f"\n--- PyTorch CPU ---")
        results = benchmark_pytorch(model, input_size, args.num_runs, args.warmup, 'cpu')
        print(f"Mean: {results['mean_ms']:.2f} ± {results['std_ms']:.2f} ms")
        print(f"Min/Max: {results['min_ms']:.2f} / {results['max_ms']:.2f} ms")
        print(f"FPS: {results['fps']:.1f}")
        
        # PyTorch GPU
        if torch.cuda.is_available() and args.gpu:
            print(f"\n--- PyTorch GPU ({torch.cuda.get_device_name(0)}) ---")
            results = benchmark_pytorch(model, input_size, args.num_runs, args.warmup, 'cuda')
            print(f"Mean: {results['mean_ms']:.2f} ± {results['std_ms']:.2f} ms")
            print(f"Min/Max: {results['min_ms']:.2f} / {results['max_ms']:.2f} ms")
            print(f"FPS: {results['fps']:.1f}")
            
            print(f"\n--- Memory Usage ---")
            mem_results = benchmark_memory(model, input_size)
            print(f"Peak (FP32): {mem_results['peak_memory_mb']:.2f} MB")
            print(f"Peak (AMP): {mem_results['peak_memory_amp_mb']:.2f} MB")
            print(f"Reduction: {mem_results['reduction_pct']:.1f}%")
    
    except ImportError as e:
        print(f"Could not import model: {e}")
    
    # ONNX Runtime
    if Path(args.model_path).exists() and ONNX_AVAILABLE:
        print(f"\n--- ONNX Runtime CPU ---")
        results = benchmark_onnx(args.model_path, input_size, args.num_runs, args.warmup, False)
        print(f"Provider: {results.get('provider', 'N/A')}")
        print(f"Mean: {results['mean_ms']:.2f} ± {results['std_ms']:.2f} ms")
        print(f"FPS: {results['fps']:.1f}")
        
        if args.gpu:
            print(f"\n--- ONNX Runtime GPU ---")
            results = benchmark_onnx(args.model_path, input_size, args.num_runs, args.warmup, True)
            print(f"Provider: {results.get('provider', 'N/A')}")
            print(f"Mean: {results['mean_ms']:.2f} ± {results['std_ms']:.2f} ms")
            print(f"FPS: {results['fps']:.1f}")
    
    print("\n" + "=" * 60)
    print("Benchmark Complete")
    print("=" * 60)


if __name__ == '__main__':
    main()
