#!/usr/bin/env python3
"""Benchmark ReID pipeline per-stage latency, GPU memory, and accuracy metrics.

Usage:
    python scripts/benchmark-reid-pipeline-performance.py --video input.mp4
    python scripts/benchmark-reid-pipeline-performance.py --video input.mp4 --warmup 50 --frames 300
    python scripts/benchmark-reid-pipeline-performance.py --video input.mp4 --output outputs/benchmarks/
"""
import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reid_research import ReIDConfig, VideoReIDPipeline, load_config


# --- GPU-accurate timing ---

class GPUTimer:
    """Context manager for GPU-synchronized timing via CUDA events."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self.elapsed_ms = 0.0
        if self.enabled:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if self.enabled:
            self.start_event.record()
        else:
            self._wall_start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.enabled:
            self.end_event.record()
            torch.cuda.synchronize()
            self.elapsed_ms = self.start_event.elapsed_time(self.end_event)
        else:
            self.elapsed_ms = (time.perf_counter() - self._wall_start) * 1000.0


@dataclass
class FrameMetrics:
    """Per-frame timing and accuracy data."""
    detection_ms: float = 0.0
    extraction_ms: float = 0.0
    matching_ms: float = 0.0
    visualization_ms: float = 0.0
    total_ms: float = 0.0
    num_detections: int = 0
    num_matches: int = 0
    unique_ids: int = 0
    similarities: list[float] = field(default_factory=list)


def compute_stats(values: list[float]) -> dict:
    """Compute statistical summary for a list of values."""
    if not values:
        return {"mean": 0, "std": 0, "p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0}
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def get_system_info() -> dict:
    """Collect system and GPU information."""
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_total_mb"] = torch.cuda.get_device_properties(0).total_mem / 1e6
        info["cuda_version"] = torch.version.cuda
    return info


def get_gpu_memory() -> dict:
    """Capture current GPU memory usage."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "reserved_mb": 0, "peak_mb": 0}
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1e6,
        "reserved_mb": torch.cuda.memory_reserved() / 1e6,
        "peak_mb": torch.cuda.max_memory_allocated() / 1e6,
    }


def benchmark_pipeline(
    pipeline: VideoReIDPipeline,
    video_path: Path,
    warmup_frames: int = 50,
    measure_frames: int = 300,
) -> dict:
    """Run benchmark on pipeline with per-stage GPU timing.

    Args:
        pipeline: Initialized VideoReIDPipeline
        video_path: Path to input video
        warmup_frames: Frames to skip before measurement
        measure_frames: Frames to measure

    Returns:
        Benchmark results dict
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_available = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    resolution = f"{width}x{height}"
    use_gpu = torch.cuda.is_available()

    # Cold-start timing: measure from script start to first inference result
    cold_start_time = time.perf_counter()

    # Memory before
    mem_before = get_gpu_memory()
    if use_gpu:
        torch.cuda.reset_peak_memory_stats()

    # --- Warmup phase ---
    print(f"Warming up ({warmup_frames} frames)...")
    for _ in range(warmup_frames):
        ret, frame = cap.read()
        if not ret:
            break
        pipeline.process_frame(frame)
        pipeline.gallery.step_frame()

    # Record cold start (includes model load + warmup)
    cold_start_ms = (time.perf_counter() - cold_start_time) * 1000.0

    # --- Measurement phase ---
    print(f"Measuring ({measure_frames} frames @ {resolution})...")
    frame_metrics: list[FrameMetrics] = []
    all_unique_ids: set[int] = set()
    id_switches = 0
    last_assignments: dict[int, int] = {}  # det_position_hash -> track_id

    for frame_idx in range(measure_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Video ended at frame {frame_idx}")
            break

        metrics = FrameMetrics()

        # Stage 0: Detection
        with GPUTimer(use_gpu) as t_det:
            detections = pipeline.detector.detect(frame)
        metrics.detection_ms = t_det.elapsed_ms
        metrics.num_detections = len(detections)

        if detections:
            # Stage 1: Feature extraction
            crops = [d.crop for d in detections]
            with GPUTimer(use_gpu) as t_ext:
                features = pipeline.extractor.extract(crops)
            metrics.extraction_ms = t_ext.elapsed_ms

            # Assign features
            for det, feat in zip(detections, features):
                det.features = feat

            # Stage 2: Matching
            features_list = [d.features for d in detections]
            bboxes = [d.bbox for d in detections]
            with GPUTimer(use_gpu) as t_match:
                results = pipeline.gallery.match_batch(features_list, bboxes=bboxes)
            metrics.matching_ms = t_match.elapsed_ms

            # Count matches and collect similarities
            for (match_id, sim), det in zip(results, detections):
                if match_id is not None:
                    metrics.num_matches += 1
                    metrics.similarities.append(sim)
                    all_unique_ids.add(match_id)

                    # Track ID switches (position-based)
                    pos_hash = hash((int(det.bbox[0] // 50), int(det.bbox[1] // 50)))
                    if pos_hash in last_assignments and last_assignments[pos_hash] != match_id:
                        id_switches += 1
                    last_assignments[pos_hash] = match_id

        # Stage 3: Visualization (measure even if no detections)
        with GPUTimer(use_gpu) as t_viz:
            if detections:
                # Simulate visualization work without writing
                _ = pipeline._visualize(frame, detections)
        metrics.visualization_ms = t_viz.elapsed_ms

        metrics.total_ms = (
            metrics.detection_ms + metrics.extraction_ms
            + metrics.matching_ms + metrics.visualization_ms
        )
        metrics.unique_ids = len(all_unique_ids)

        pipeline.gallery.step_frame()
        frame_metrics.append(metrics)

        if (frame_idx + 1) % 50 == 0:
            avg_total = np.mean([m.total_ms for m in frame_metrics[-50:]])
            print(f"  Frame {frame_idx + 1}/{measure_frames}: avg {avg_total:.1f}ms ({1000/avg_total:.0f} FPS)")

    cap.release()

    # Memory after
    mem_after = get_gpu_memory()

    # --- Compile results ---
    detection_times = [m.detection_ms for m in frame_metrics]
    extraction_times = [m.extraction_ms for m in frame_metrics]
    matching_times = [m.matching_ms for m in frame_metrics]
    viz_times = [m.visualization_ms for m in frame_metrics]
    total_times = [m.total_ms for m in frame_metrics]
    all_sims = [s for m in frame_metrics for s in m.similarities]
    det_counts = [m.num_detections for m in frame_metrics]

    results = {
        "system": get_system_info(),
        "video": {
            "path": str(video_path),
            "resolution": resolution,
            "fps": fps,
            "total_frames": total_available,
        },
        "benchmark_params": {
            "warmup_frames": warmup_frames,
            "measure_frames": len(frame_metrics),
        },
        "cold_start_ms": cold_start_ms,
        "latency": {
            "detection": compute_stats(detection_times),
            "extraction": compute_stats(extraction_times),
            "matching": compute_stats(matching_times),
            "visualization": compute_stats(viz_times),
            "total": compute_stats(total_times),
        },
        "throughput": {
            "mean_fps": 1000.0 / np.mean(total_times) if total_times else 0,
            "p50_fps": 1000.0 / np.percentile(total_times, 50) if total_times else 0,
            "p99_fps": 1000.0 / np.percentile(total_times, 99) if total_times else 0,
        },
        "memory": {
            "before": mem_before,
            "after": mem_after,
            "peak_mb": mem_after.get("peak_mb", 0),
        },
        "accuracy": {
            "total_detections": sum(det_counts),
            "mean_detections_per_frame": float(np.mean(det_counts)),
            "total_matches": sum(m.num_matches for m in frame_metrics),
            "unique_ids": len(all_unique_ids),
            "id_switches": id_switches,
            "similarity_distribution": compute_stats(all_sims) if all_sims else {},
        },
    }

    return results


def generate_markdown_report(results: dict) -> str:
    """Generate human-readable markdown report from benchmark results."""
    lat = results["latency"]
    tp = results["throughput"]
    acc = results["accuracy"]
    mem = results["memory"]
    sys_info = results["system"]

    report = f"""# ReID Pipeline Benchmark Report

## System
- **GPU:** {sys_info.get('gpu_name', 'N/A')}
- **PyTorch:** {sys_info.get('pytorch', 'N/A')}
- **CUDA:** {sys_info.get('cuda_version', 'N/A')}

## Video
- **Resolution:** {results['video']['resolution']}
- **FPS:** {results['video']['fps']}
- **Measured Frames:** {results['benchmark_params']['measure_frames']}

## Cold Start
- **Time to first inference:** {results['cold_start_ms']:.0f}ms

## Per-Stage Latency (ms)

| Stage | Mean | Std | P50 | P95 | P99 |
|-------|------|-----|-----|-----|-----|
| Detection | {lat['detection']['mean']:.1f} | {lat['detection']['std']:.1f} | {lat['detection']['p50']:.1f} | {lat['detection']['p95']:.1f} | {lat['detection']['p99']:.1f} |
| Extraction | {lat['extraction']['mean']:.1f} | {lat['extraction']['std']:.1f} | {lat['extraction']['p50']:.1f} | {lat['extraction']['p95']:.1f} | {lat['extraction']['p99']:.1f} |
| Matching | {lat['matching']['mean']:.1f} | {lat['matching']['std']:.1f} | {lat['matching']['p50']:.1f} | {lat['matching']['p95']:.1f} | {lat['matching']['p99']:.1f} |
| Visualization | {lat['visualization']['mean']:.1f} | {lat['visualization']['std']:.1f} | {lat['visualization']['p50']:.1f} | {lat['visualization']['p95']:.1f} | {lat['visualization']['p99']:.1f} |
| **Total** | **{lat['total']['mean']:.1f}** | {lat['total']['std']:.1f} | **{lat['total']['p50']:.1f}** | {lat['total']['p95']:.1f} | {lat['total']['p99']:.1f} |

## Throughput

| Metric | FPS |
|--------|-----|
| Mean | {tp['mean_fps']:.1f} |
| P50 | {tp['p50_fps']:.1f} |
| P99 | {tp['p99_fps']:.1f} |

## Memory

| Metric | MB |
|--------|------|
| Peak GPU | {mem.get('peak_mb', 0):.0f} |
| Allocated (after) | {mem['after'].get('allocated_mb', 0):.0f} |

## Accuracy

| Metric | Value |
|--------|-------|
| Total Detections | {acc['total_detections']} |
| Mean Det/Frame | {acc['mean_detections_per_frame']:.1f} |
| Unique IDs | {acc['unique_ids']} |
| ID Switches | {acc['id_switches']} |
| Match Similarity (mean) | {acc['similarity_distribution'].get('mean', 0):.3f} |
"""
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ReID pipeline performance")
    parser.add_argument("--video", "-v", type=str, default=None, help="Input video path")
    parser.add_argument("--config", "-c", type=str, default="configs/default.yaml", help="Config file")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup frames (default: 50)")
    parser.add_argument("--frames", type=int, default=300, help="Frames to measure (default: 300)")
    parser.add_argument("--output", "-o", type=str, default="outputs/benchmarks/", help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Load config
    config_path = Path(args.config)
    config = load_config(config_path) if config_path.exists() else ReIDConfig()
    if args.device:
        config.model.device = args.device

    # Disable video output for benchmarking
    config.output.save_video = False
    config.output.visualization = False

    # Find video
    if args.video:
        video_path = Path(args.video)
    else:
        video_dir = Path("data/videos")
        videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
        if not videos:
            print(f"Error: No videos found in {video_dir}")
            return 1
        video_path = videos[0]
        print(f"Auto-selected: {video_path}")

    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return 1

    # Create pipeline
    print("Initializing pipeline...")
    pipeline = VideoReIDPipeline(config)

    # Run benchmark
    results = benchmark_pipeline(
        pipeline, video_path,
        warmup_frames=args.warmup,
        measure_frames=args.frames,
    )

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    resolution_tag = results["video"]["resolution"].replace("x", "p")

    json_path = output_dir / f"benchmark-{timestamp}-{resolution_tag}.json"
    md_path = output_dir / f"benchmark-{timestamp}-{resolution_tag}.md"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON report: {json_path}")

    md_report = generate_markdown_report(results)
    md_path.write_text(md_report)
    print(f"Markdown report: {md_path}")

    # Print summary
    lat = results["latency"]["total"]
    tp = results["throughput"]
    print(f"\n{'='*50}")
    print(f"BENCHMARK SUMMARY ({results['video']['resolution']})")
    print(f"{'='*50}")
    print(f"Mean latency: {lat['mean']:.1f}ms | Mean FPS: {tp['mean_fps']:.1f}")
    print(f"P99 latency:  {lat['p99']:.1f}ms | P99 FPS:  {tp['p99_fps']:.1f}")
    print(f"Cold start:   {results['cold_start_ms']:.0f}ms")
    print(f"Peak GPU mem: {results['memory'].get('peak_mb', 0):.0f}MB")
    print(f"Unique IDs:   {results['accuracy']['unique_ids']}")
    print(f"ID switches:  {results['accuracy']['id_switches']}")
    print(f"{'='*50}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
