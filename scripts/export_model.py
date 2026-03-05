"""
Export MobileFaceNet to ONNX and TorchScript

Exports the trained backbone for production deployment:
  - ONNX (opset 14, dynamic batch) for cross-platform inference
  - TorchScript (torch.jit.trace) for PyTorch deployment
  - Numerical verification (PyTorch vs ONNX output)
  - Inference speed benchmarking (PyTorch vs ONNX Runtime)

Usage:
    python scripts/export_model.py
    python scripts/export_model.py --model-path models/checkpoints/best_backbone.pth
    python scripts/export_model.py --output-dir models/exported --opset 14

Requirements:
    pip install torch onnx onnxruntime
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Environment Detection
# ---------------------------------------------------------------------------
try:
    from google.colab import drive  # noqa: F401

    IN_COLAB = True
    PROJECT_ROOT = Path("/content/face_recognition")
except ImportError:
    IN_COLAB = False
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))
from scripts.model import MobileFaceNet  # noqa: E402


# =============================================================================
# Model Loading
# =============================================================================

def load_backbone(model_path, emb_dim=512):
    """Load MobileFaceNet backbone from checkpoint or state dict."""
    model = MobileFaceNet(emb_dim=emb_dim)
    model_path = Path(model_path)

    if not model_path.exists():
        print(f"  Model not found: {model_path}")
        print(f"  Using random weights")
        model.eval()
        return model

    state = torch.load(model_path, map_location="cpu")

    if isinstance(state, dict) and "model" in state:
        backbone_state = {
            k.replace("backbone.", ""): v
            for k, v in state["model"].items()
            if k.startswith("backbone.")
        }
        if backbone_state:
            model.load_state_dict(backbone_state)
            print(f"  Loaded backbone from checkpoint: {model_path.name}")
        else:
            model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
        print(f"  Loaded model: {model_path.name}")

    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    return model


# =============================================================================
# ONNX Export
# =============================================================================

def export_to_onnx(model, output_path, input_shape=(1, 3, 112, 112), opset_version=14):
    """
    Export PyTorch model to ONNX format.

    Uses dynamo=False (legacy TorchScript-based tracer) for broad compatibility
    with export_params, dynamic_axes, and older ONNX Runtime versions.
    """
    import onnx

    model.eval()
    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
        dynamo=False,  # Legacy tracer for compatibility
    )

    # Verify
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  ONNX exported: {output_path}")
    print(f"    Size: {size_mb:.2f} MB")
    print(f"    Opset: {opset_version}")
    print(f"    Dynamic batch: yes")

    return output_path


# =============================================================================
# TorchScript Export
# =============================================================================

def export_to_torchscript(model, output_path, input_shape=(1, 3, 112, 112)):
    """Export model to TorchScript via tracing."""
    model.eval()
    dummy_input = torch.randn(*input_shape)

    traced = torch.jit.trace(model, dummy_input)
    traced.save(str(output_path))

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  TorchScript exported: {output_path}")
    print(f"    Size: {size_mb:.2f} MB")

    return output_path


# =============================================================================
# Verification
# =============================================================================

def verify_onnx(onnx_path, pytorch_model):
    """Verify ONNX model produces identical output to PyTorch."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))

    test_input = np.random.randn(1, 3, 112, 112).astype(np.float32)

    # PyTorch output
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_out = pytorch_model(torch.from_numpy(test_input)).numpy()

    # ONNX output
    onnx_out = session.run(None, {"input": test_input})[0]

    max_diff = np.abs(pytorch_out - onnx_out).max()
    mean_diff = np.abs(pytorch_out - onnx_out).mean()

    print(f"\n  ONNX Verification:")
    print(f"    Max difference:  {max_diff:.8f}")
    print(f"    Mean difference: {mean_diff:.8f}")

    ok = max_diff < 1e-4
    print(f"    Status: {'PASSED' if ok else 'NUMERICAL DIFFERENCES DETECTED'}")
    return ok


# =============================================================================
# Benchmarking
# =============================================================================

def benchmark_onnx(onnx_path, num_iterations=100, batch_sizes=(1, 8, 32)):
    """Benchmark ONNX Runtime inference speed."""
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), session_options)

    print(f"\n  ONNX Runtime Benchmark:")
    print(f"  {'=' * 45}")

    results = {}
    for bs in batch_sizes:
        test_input = np.random.randn(bs, 3, 112, 112).astype(np.float32)

        # Warmup
        for _ in range(10):
            _ = session.run(None, {"input": test_input})

        start = time.time()
        for _ in range(num_iterations):
            _ = session.run(None, {"input": test_input})
        total = time.time() - start

        avg_ms = total / num_iterations * 1000
        throughput = bs / (avg_ms / 1000)
        results[bs] = (avg_ms, throughput)
        print(f"    Batch {bs:>2}: {avg_ms:.2f} ms ({throughput:.1f} img/s)")

    print(f"  {'=' * 45}")
    return results


def benchmark_pytorch(model, num_iterations=100, batch_sizes=(1, 8, 32)):
    """Benchmark PyTorch inference speed (CPU)."""
    model.eval()

    print(f"\n  PyTorch Benchmark:")
    print(f"  {'=' * 45}")

    results = {}
    for bs in batch_sizes:
        test_input = torch.randn(bs, 3, 112, 112)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(test_input)

        start = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(test_input)
        total = time.time() - start

        avg_ms = total / num_iterations * 1000
        throughput = bs / (avg_ms / 1000)
        results[bs] = (avg_ms, throughput)
        print(f"    Batch {bs:>2}: {avg_ms:.2f} ms ({throughput:.1f} img/s)")

    print(f"  {'=' * 45}")
    return results


def compare_benchmarks(pytorch_results, onnx_results):
    """Print side-by-side PyTorch vs ONNX comparison."""
    print(f"\n  PyTorch vs ONNX Comparison:")
    print(f"  {'=' * 55}")
    print(f"  {'Batch':>5} | {'PyTorch':>10} | {'ONNX':>10} | {'Speedup':>8}")
    print(f"  {'-' * 55}")

    for bs in pytorch_results:
        pt_ms = pytorch_results[bs][0]
        ox_ms = onnx_results[bs][0]
        speedup = pt_ms / ox_ms
        print(f"  {bs:>5} | {pt_ms:>8.2f}ms | {ox_ms:>8.2f}ms | {speedup:>6.2f}x")

    print(f"  {'=' * 55}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Export MobileFaceNet to ONNX/TorchScript")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model weights (default: models/checkpoints/best_backbone.pth)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Export output directory (default: models/exported)")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--skip-torchscript", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--benchmark-iterations", type=int, default=100)
    args = parser.parse_args()

    # Paths
    checkpoint_dir = PROJECT_ROOT / "models" / "checkpoints"
    model_path = Path(args.model_path) if args.model_path else checkpoint_dir / "best_backbone.pth"
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "models" / "exported"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"MobileFaceNet Model Export")
    print(f"{'=' * 60}")

    # Load model
    model = load_backbone(model_path, args.embedding_dim)

    # Export ONNX
    onnx_path = export_to_onnx(model, output_dir / "mobilefacenet.onnx", opset_version=args.opset)

    # Verify ONNX
    verify_onnx(onnx_path, model)

    # Export TorchScript
    if not args.skip_torchscript:
        export_to_torchscript(model, output_dir / "mobilefacenet.pt")

    # Benchmark
    if not args.skip_benchmark:
        pt_results = benchmark_pytorch(model, args.benchmark_iterations)
        ox_results = benchmark_onnx(onnx_path, args.benchmark_iterations)
        compare_benchmarks(pt_results, ox_results)

    # List exported files
    print(f"\n  Exported Files:")
    print(f"  {'=' * 45}")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size / 1024
        unit = "KB"
        if size > 1024:
            size /= 1024
            unit = "MB"
        print(f"    {f.name}: {size:.1f} {unit}")
    print(f"  {'=' * 45}")

    print(f"\n  Export complete! Files in {output_dir}")
    print(f"  Use models/exported/inference.py for deployment.")


if __name__ == "__main__":
    main()
