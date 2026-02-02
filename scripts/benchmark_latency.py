#!/usr/bin/env python3
"""Benchmark component latencies for HAT-ReID pipeline.

Usage:
    python scripts/benchmark_latency.py
    python scripts/benchmark_latency.py --device cpu
"""

import argparse
import sys
import time
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hat_reid import LDA, get_device
from src.embeddings import EmbeddingExtractor


def benchmark_lda(
    n_samples: int = 500,
    n_classes: int = 20,
    dim: int = 256,
    n_runs: int = 100,
    device: str = "cuda",
) -> dict:
    """Benchmark LDA fit + transform latency.

    Args:
        n_samples: Number of history samples
        n_classes: Number of track classes
        dim: Feature dimension
        n_runs: Number of benchmark iterations
        device: Computation device

    Returns:
        Dict with timing results
    """
    X = torch.randn(n_samples, dim, device=device)
    y = torch.randint(0, n_classes, (n_samples,), device=device)
    X_test = torch.randn(50, dim, device=device)

    # Warmup
    lda = LDA(device=device)
    lda.fit(X, y)
    lda.transform(X_test)

    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    for _ in range(n_runs):
        lda = LDA(device=device)
        lda.fit(X, y)
        lda.transform(X_test)

    if device == "cuda":
        torch.cuda.synchronize()

    total_ms = (time.perf_counter() - t0) * 1000
    avg_ms = total_ms / n_runs

    return {
        "component": "LDA fit+transform",
        "n_samples": n_samples,
        "n_classes": n_classes,
        "dim": dim,
        "n_runs": n_runs,
        "total_ms": total_ms,
        "avg_ms": avg_ms,
    }


def benchmark_extractor(
    batch_sizes: list[int] = [1, 5, 10, 20],
    n_runs: int = 50,
    device: str = "cuda",
) -> list[dict]:
    """Benchmark embedding extraction latency.

    Args:
        batch_sizes: List of batch sizes to test
        n_runs: Number of benchmark iterations
        device: Computation device

    Returns:
        List of dicts with timing results
    """
    ext = EmbeddingExtractor(model_name="resnet50", device=device)
    results = []

    for bs in batch_sizes:
        crops = torch.randn(bs, 3, 256, 128, device=device)

        # Warmup
        ext.extract(crops)

        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        for _ in range(n_runs):
            ext.extract(crops)

        if device == "cuda":
            torch.cuda.synchronize()

        total_ms = (time.perf_counter() - t0) * 1000
        avg_ms = total_ms / n_runs

        results.append({
            "component": f"Embedding (batch={bs})",
            "batch_size": bs,
            "n_runs": n_runs,
            "total_ms": total_ms,
            "avg_ms": avg_ms,
        })

    return results


def benchmark_tracker_update(
    n_detections: int = 20,
    n_tracks: int = 10,
    dim: int = 256,
    n_runs: int = 100,
    device: str = "cuda",
) -> dict:
    """Benchmark tracker update latency.

    Args:
        n_detections: Number of detections per frame
        n_tracks: Number of existing tracks
        dim: Embedding dimension
        n_runs: Number of benchmark iterations
        device: Computation device

    Returns:
        Dict with timing results
    """
    from src.tracker import HATTracker

    tracker = HATTracker(device=device, use_hat=True)

    # Build up some track history
    for i in range(30):
        boxes = torch.rand(n_tracks, 4, device=device) * 500
        boxes[:, 2:] += boxes[:, :2] + 50
        scores = torch.rand(n_tracks, device=device) * 0.3 + 0.7
        embeds = torch.randn(n_tracks, dim, device=device)
        tracker.update(boxes, scores, embeds, frame_id=i)

    # Prepare test data
    boxes = torch.rand(n_detections, 4, device=device) * 500
    boxes[:, 2:] += boxes[:, :2] + 50
    scores = torch.rand(n_detections, device=device) * 0.3 + 0.7
    embeds = torch.randn(n_detections, dim, device=device)

    # Warmup
    tracker.update(boxes, scores, embeds, frame_id=100)

    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    for i in range(n_runs):
        tracker.update(boxes, scores, embeds, frame_id=100 + i)

    if device == "cuda":
        torch.cuda.synchronize()

    total_ms = (time.perf_counter() - t0) * 1000
    avg_ms = total_ms / n_runs

    return {
        "component": "Tracker update",
        "n_detections": n_detections,
        "n_tracks": n_tracks,
        "n_runs": n_runs,
        "total_ms": total_ms,
        "avg_ms": avg_ms,
        "hat_active": tracker.hat_active,
    }


def print_results(results: list[dict]) -> None:
    """Print benchmark results in table format."""
    print("\n" + "=" * 60)
    print("HAT-ReID Benchmark Results")
    print("=" * 60)

    for r in results:
        component = r.get("component", "Unknown")
        avg_ms = r.get("avg_ms", 0)
        n_runs = r.get("n_runs", 0)

        print(f"\n{component}")
        print("-" * 40)
        print(f"  Average latency: {avg_ms:.2f}ms")
        print(f"  Iterations: {n_runs}")

        # Component-specific details
        if "n_samples" in r:
            print(f"  Samples: {r['n_samples']}, Classes: {r['n_classes']}, Dim: {r['dim']}")
        if "batch_size" in r:
            print(f"  Batch size: {r['batch_size']}")
        if "n_detections" in r:
            print(f"  Detections: {r['n_detections']}, Tracks: {r['n_tracks']}")
        if "hat_active" in r:
            print(f"  HAT active: {r['hat_active']}")

    print("\n" + "=" * 60)

    # Summary table
    print("\nSummary (avg latency):")
    print("-" * 40)
    for r in results:
        print(f"  {r['component']:<30} {r['avg_ms']:>8.2f}ms")


def main():
    parser = argparse.ArgumentParser(description="HAT-ReID Latency Benchmark")
    parser.add_argument("--device", default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--lda-only", action="store_true",
                        help="Only benchmark LDA")
    parser.add_argument("--extractor-only", action="store_true",
                        help="Only benchmark extractor")
    parser.add_argument("--tracker-only", action="store_true",
                        help="Only benchmark tracker")

    args = parser.parse_args()

    # Validate device
    if args.device == "cuda":
        try:
            device = str(get_device())
            print(f"Using device: {device}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        except RuntimeError as e:
            print(f"CUDA error: {e}")
            print("Falling back to CPU")
            args.device = "cpu"
    else:
        print(f"Using device: {args.device}")

    results = []

    run_all = not (args.lda_only or args.extractor_only or args.tracker_only)

    # LDA benchmark
    if run_all or args.lda_only:
        print("\nBenchmarking LDA...")
        lda_result = benchmark_lda(device=args.device)
        results.append(lda_result)

    # Extractor benchmark
    if run_all or args.extractor_only:
        print("\nBenchmarking Embedding Extractor...")
        ext_results = benchmark_extractor(device=args.device)
        results.extend(ext_results)

    # Tracker benchmark
    if run_all or args.tracker_only:
        print("\nBenchmarking Tracker...")
        tracker_result = benchmark_tracker_update(device=args.device)
        results.append(tracker_result)

    print_results(results)


if __name__ == "__main__":
    main()
