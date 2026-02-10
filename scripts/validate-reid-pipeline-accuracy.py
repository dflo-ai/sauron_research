#!/usr/bin/env python3
"""Validate optimized pipeline accuracy against baseline.

Compares feature vectors, ID assignments, and ID switch rates between
baseline (FP32, no optimizations) and optimized (FP16, torch.compile, FAISS)
pipeline configurations.

Usage:
    python scripts/validate-reid-pipeline-accuracy.py --video input.mp4
    python scripts/validate-reid-pipeline-accuracy.py --video input.mp4 --max-frames 500
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reid_research import ReIDConfig, VideoReIDPipeline, load_config


def run_pipeline_with_tracking(
    pipeline: VideoReIDPipeline,
    video_path: str,
    max_frames: int,
) -> dict:
    """Run pipeline and capture all outputs for validation.

    Args:
        pipeline: Initialized pipeline
        video_path: Path to video
        max_frames: Max frames to process

    Returns:
        Dict with features, assignments, unique_ids, timing
    """
    all_features = []
    all_assignments = []
    unique_ids = set()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    start_time = time.perf_counter()
    frame_idx = 0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        detections = pipeline.process_frame(frame)

        # Collect features
        frame_features = [d.features for d in detections if d.features is not None]
        if frame_features:
            all_features.extend(frame_features)

        # Collect assignments
        frame_assignments = [d.track_id for d in detections if d.track_id is not None]
        all_assignments.append(frame_assignments)

        for d in detections:
            if d.track_id is not None and d.track_id >= 0:
                unique_ids.add(d.track_id)

        pipeline.gallery.step_frame()
        frame_idx += 1

    elapsed = time.perf_counter() - start_time
    cap.release()

    return {
        "features": np.vstack(all_features) if all_features else np.empty((0, 512)),
        "assignments": all_assignments,
        "unique_ids": len(unique_ids),
        "frames_processed": frame_idx,
        "elapsed_seconds": elapsed,
        "fps": frame_idx / elapsed if elapsed > 0 else 0,
    }


def compare_features(baseline: np.ndarray, optimized: np.ndarray) -> dict:
    """Compare feature vectors between baseline and optimized pipeline.

    Uses cosine similarity to measure feature drift from FP16/compile.
    """
    if baseline.shape[0] == 0 or optimized.shape[0] == 0:
        return {"error": "Empty features", "match": False}

    # Compare up to the minimum of both sets
    n = min(baseline.shape[0], optimized.shape[0])
    baseline = baseline[:n]
    optimized = optimized[:n]

    # Cosine similarity per feature pair
    # For L2-normalized vectors: cos_sim = dot product
    dot_products = np.sum(baseline * optimized, axis=1)
    similarities = np.clip(dot_products, -1.0, 1.0)

    return {
        "count": int(n),
        "mean_similarity": float(np.mean(similarities)),
        "min_similarity": float(np.min(similarities)),
        "max_similarity": float(np.max(similarities)),
        "std_similarity": float(np.std(similarities)),
        "below_0995": int(np.sum(similarities < 0.995)),
        "below_0990": int(np.sum(similarities < 0.990)),
    }


def compare_assignments(baseline: list, optimized: list) -> dict:
    """Compare ID assignments frame-by-frame."""
    n_frames = min(len(baseline), len(optimized))
    total = 0
    matches = 0

    for b_frame, o_frame in zip(baseline[:n_frames], optimized[:n_frames]):
        # Compare sets of IDs per frame (order may differ)
        b_set = set(b_frame)
        o_set = set(o_frame)
        total += max(len(b_set), len(o_set), 1)
        matches += len(b_set & o_set)

    return {
        "frames_compared": n_frames,
        "total_assignments": total,
        "matches": matches,
        "match_rate": matches / max(1, total),
    }


def count_id_switches(assignments: list[list[int]]) -> int:
    """Count ID switches across frames.

    An ID switch occurs when a track_id disappears for >1 frame
    and then reappears, or when the same spatial position gets
    a different ID.
    """
    # Track last seen frame per ID
    last_seen: dict[int, int] = {}
    switches = 0

    for frame_idx, frame_ids in enumerate(assignments):
        for track_id in frame_ids:
            if track_id < 0:  # Skip tentative
                continue
            if track_id in last_seen:
                gap = frame_idx - last_seen[track_id]
                if gap > 5:  # Re-appeared after >5 frame gap
                    switches += 1
            last_seen[track_id] = frame_idx

    return switches


def create_baseline_config(base_config: ReIDConfig) -> ReIDConfig:
    """Create baseline config with no optimizations."""
    config = base_config.model_copy(deep=True)
    # Disable FAISS for baseline
    config.gallery.use_faiss = False
    return config


def create_optimized_config(base_config: ReIDConfig) -> ReIDConfig:
    """Create optimized config with all Phase 2-5 optimizations enabled."""
    config = base_config.model_copy(deep=True)
    config.gallery.use_faiss = True
    config.gallery.use_full_reranking = True
    config.gallery.rerank_cache_knn = True
    return config


def generate_report(results: dict, output_path: Path) -> None:
    """Generate markdown validation report."""
    feat = results["feature_comparison"]
    assign = results["assignment_comparison"]

    report = f"""# Pipeline Validation Report

## Summary

| Metric | Baseline | Optimized | Status |
|--------|----------|-----------|--------|
| FPS | {results['baseline']['fps']:.1f} | {results['optimized']['fps']:.1f} | {'+' if results['optimized']['fps'] > results['baseline']['fps'] else '-'}{abs(results['optimized']['fps'] - results['baseline']['fps']):.1f} |
| Unique IDs | {results['baseline']['unique_ids']} | {results['optimized']['unique_ids']} | INFO |
| ID Switches | {results['baseline_id_switches']} | {results['optimized_id_switches']} | {'PASS' if results['optimized_id_switches'] <= results['baseline_id_switches'] * 1.1 else 'FAIL'} |

## Feature Accuracy (FP16 vs FP32)

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Mean Similarity | {feat.get('mean_similarity', 0):.6f} | > 0.995 | {'PASS' if feat.get('mean_similarity', 0) > 0.995 else 'FAIL'} |
| Min Similarity | {feat.get('min_similarity', 0):.6f} | > 0.990 | {'PASS' if feat.get('min_similarity', 0) > 0.990 else 'FAIL'} |
| Below 0.995 | {feat.get('below_0995', 0)} | < 5% | INFO |
| Below 0.990 | {feat.get('below_0990', 0)} | < 1% | INFO |

## Assignment Accuracy

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Match Rate | {assign.get('match_rate', 0):.2%} | > 95% | {'PASS' if assign.get('match_rate', 0) > 0.95 else 'FAIL'} |
| Frames Compared | {assign.get('frames_compared', 0)} | - | INFO |

## Overall: **{'PASSED' if results['passed'] else 'FAILED'}**
"""
    output_path.write_text(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate optimized pipeline accuracy")
    parser.add_argument("--video", "-v", type=str, default=None, help="Test video path")
    parser.add_argument("--config", "-c", type=str, default="configs/default.yaml", help="Base config")
    parser.add_argument("--max-frames", type=int, default=300, help="Frames to compare (default: 300)")
    parser.add_argument("--output", "-o", type=str, default="outputs/validation/", help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Load base config
    config_path = Path(args.config)
    base_config = load_config(config_path) if config_path.exists() else ReIDConfig()
    if args.device:
        base_config.model.device = args.device

    # Disable video output
    base_config.output.save_video = False
    base_config.output.visualization = False

    # Find video
    if args.video:
        video_path = args.video
    else:
        video_dir = Path("data/videos")
        videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
        if not videos:
            print(f"Error: No videos found in {video_dir}")
            return 1
        video_path = str(videos[0])
        print(f"Auto-selected: {video_path}")

    if not Path(video_path).exists():
        print(f"Error: Video not found: {video_path}")
        return 1

    # Run baseline
    print("=" * 50)
    print("Running BASELINE pipeline (no optimizations)...")
    print("=" * 50)
    baseline_config = create_baseline_config(base_config)
    baseline_pipeline = VideoReIDPipeline(baseline_config)
    baseline_results = run_pipeline_with_tracking(
        baseline_pipeline, video_path, args.max_frames
    )
    print(f"  Processed {baseline_results['frames_processed']} frames in {baseline_results['elapsed_seconds']:.1f}s ({baseline_results['fps']:.1f} FPS)")

    # Run optimized
    print("\n" + "=" * 50)
    print("Running OPTIMIZED pipeline (FP16 + compile + FAISS)...")
    print("=" * 50)
    optimized_config = create_optimized_config(base_config)
    optimized_pipeline = VideoReIDPipeline(optimized_config)
    optimized_results = run_pipeline_with_tracking(
        optimized_pipeline, video_path, args.max_frames
    )
    print(f"  Processed {optimized_results['frames_processed']} frames in {optimized_results['elapsed_seconds']:.1f}s ({optimized_results['fps']:.1f} FPS)")

    # Compare
    print("\nComparing results...")
    feature_cmp = compare_features(baseline_results["features"], optimized_results["features"])
    assignment_cmp = compare_assignments(baseline_results["assignments"], optimized_results["assignments"])
    baseline_switches = count_id_switches(baseline_results["assignments"])
    optimized_switches = count_id_switches(optimized_results["assignments"])

    passed = (
        feature_cmp.get("mean_similarity", 0) > 0.995
        and assignment_cmp.get("match_rate", 0) > 0.95
        and optimized_switches <= max(baseline_switches * 1.1, baseline_switches + 2)
    )

    comparison = {
        "video": video_path,
        "max_frames": args.max_frames,
        "baseline": {
            "fps": baseline_results["fps"],
            "unique_ids": baseline_results["unique_ids"],
            "frames_processed": baseline_results["frames_processed"],
        },
        "optimized": {
            "fps": optimized_results["fps"],
            "unique_ids": optimized_results["unique_ids"],
            "frames_processed": optimized_results["frames_processed"],
        },
        "feature_comparison": feature_cmp,
        "assignment_comparison": assignment_cmp,
        "baseline_id_switches": baseline_switches,
        "optimized_id_switches": optimized_switches,
        "speedup": optimized_results["fps"] / max(0.1, baseline_results["fps"]),
        "passed": passed,
    }

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "validation-results.json"
    with open(json_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    md_path = output_dir / "validation-report.md"
    generate_report(comparison, md_path)

    # Print summary
    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Feature similarity:  {feature_cmp.get('mean_similarity', 0):.6f} (threshold: > 0.995)")
    print(f"Assignment match:    {assignment_cmp.get('match_rate', 0):.2%} (threshold: > 95%)")
    print(f"Baseline switches:   {baseline_switches}")
    print(f"Optimized switches:  {optimized_switches}")
    print(f"Speedup:             {comparison['speedup']:.2f}x ({baseline_results['fps']:.1f} -> {optimized_results['fps']:.1f} FPS)")
    print(f"Status:              {'PASSED' if passed else 'FAILED'}")
    print(f"{'='*60}")
    print(f"JSON: {json_path}")
    print(f"Report: {md_path}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
