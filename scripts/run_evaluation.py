#!/usr/bin/env python3
"""Evaluate HAT-ReID tracker on video sequences.

Usage:
    python scripts/run_evaluation.py data/videos/test.mp4 --visualize
    python scripts/run_evaluation.py data/videos/test.mp4 --no-hat --output output.mp4
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hat_reid import get_device
from src.tracker import HATTracker
from src.embeddings import EmbeddingExtractor


@dataclass
class EvalMetrics:
    """Tracking evaluation metrics."""
    total_frames: int = 0
    total_detections: int = 0
    total_tracks: int = 0
    id_switches: int = 0
    hat_activations: int = 0
    total_time_ms: float = 0.0
    frame_times: list = field(default_factory=list)

    @property
    def fps(self) -> float:
        return 1000 * self.total_frames / max(self.total_time_ms, 1)

    @property
    def hat_activation_rate(self) -> float:
        return self.hat_activations / max(self.total_frames, 1)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_time_ms / max(self.total_frames, 1)

    def summary(self) -> str:
        return f"""
=== Evaluation Results ===
  Frames processed: {self.total_frames}
  Total detections: {self.total_detections}
  Unique tracks: {self.total_tracks}
  ID switches: {self.id_switches}
  HAT activations: {self.hat_activations} ({self.hat_activation_rate:.1%})
  Average FPS: {self.fps:.1f}
  Average latency: {self.avg_latency_ms:.1f}ms
"""


def get_color(track_id: int) -> tuple[int, int, int]:
    """Generate consistent color for track ID."""
    import hashlib
    h = hashlib.md5(str(track_id).encode()).hexdigest()
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def draw_tracks(
    frame,
    boxes,
    track_ids,
    hat_active: bool = False,
) -> None:
    """Draw bounding boxes with track IDs on frame (in-place)."""
    for box, tid in zip(boxes, track_ids):
        if tid < 0:
            continue
        x1, y1, x2, y2 = map(int, box)
        color = get_color(int(tid))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, f"ID:{tid}", (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    # HAT indicator
    if hat_active:
        cv2.putText(
            frame, "HAT ACTIVE", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )


def detect_persons_placeholder(frame) -> tuple[list, list]:
    """Placeholder detector - returns empty detections.

    In real usage, replace with actual YOLO or other detector.
    """
    # TODO: Integrate actual detector (YOLO, etc.)
    return [], []


# Optional: YOLO detector if ultralytics is available
_yolo_model = None

def detect_persons_yolo(frame, conf_threshold: float = 0.5) -> tuple[list, list]:
    """Detect persons using YOLO11.

    Requires: pip install ultralytics
    """
    global _yolo_model

    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            _yolo_model = YOLO("yolo11s.pt")  # Nano model for speed
        except ImportError:
            print("Warning: ultralytics not installed. Run: pip install ultralytics")
            return [], []

    results = _yolo_model(frame, verbose=False, conf=conf_threshold)

    boxes, scores = [], []
    for r in results:
        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            if int(cls) == 0:  # Person class
                boxes.append(box.cpu().numpy().tolist())
                scores.append(float(conf))

    return boxes, scores


def run_evaluation(
    video_path: str,
    config_path: str = "configs/default.yaml",
    use_hat: bool = True,
    device: str = "cuda",
    visualize: bool = False,
    output_path: str | None = None,
    max_frames: int | None = None,
) -> EvalMetrics:
    """Run tracker evaluation on video.

    Args:
        video_path: Path to input video
        config_path: Tracker config YAML
        use_hat: Enable HAT transformation
        device: cuda or cpu
        visualize: Show live visualization
        output_path: Save annotated video
        max_frames: Limit frames to process (for testing)

    Returns:
        EvalMetrics with results
    """
    # Verify CUDA availability per project requirements
    if device == "cuda":
        device_obj = get_device()
    else:
        device_obj = torch.device("cpu")

    # Load config
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f)
    else:
        print(f"Warning: Config not found at {config_path}, using defaults")
        config = {}

    # Initialize components
    hat_cfg = config.get("hat", {})
    tracker_cfg = config.get("tracker", {})
    embed_cfg = config.get("embedding", {})

    tracker = HATTracker(
        use_hat=use_hat,
        device=str(device_obj),
        hat_factor_thr=hat_cfg.get("transfer_factor_threshold", 4.0),
        history_max_len=hat_cfg.get("history_max_len", 60),
        history_decay=hat_cfg.get("history_weight_decay", 0.9),
        queue_type=hat_cfg.get("history_queue_type", "fifo"),
        match_score_thr=tracker_cfg.get("match_score_threshold", 0.5),
        init_score_thr=tracker_cfg.get("init_score_threshold", 0.8),
        max_lost_frames=tracker_cfg.get("max_lost_frames", 10),
        memo_momentum=tracker_cfg.get("memo_momentum", 0.8),
    )

    extractor = EmbeddingExtractor(
        model_name=embed_cfg.get("model", "resnet50"),
        embedding_dim=embed_cfg.get("dim", 256),
        normalize=embed_cfg.get("normalize", True),
        device=str(device_obj),
    )

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height} @ {fps:.1f} FPS")
    print(f"Total frames: {total_video_frames}")
    print(f"HAT enabled: {use_hat}")
    print()

    # Output writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    metrics = EvalMetrics()
    prev_ids = set()

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames and frame_id >= max_frames:
            break

        t0 = time.perf_counter()

        # Detection (uses YOLO if available, else placeholder)
        boxes, scores = detect_persons_yolo(frame)

        if len(boxes) > 0:
            # Convert to tensors
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_t = torch.from_numpy(frame_rgb)
            boxes_t = torch.tensor(boxes, dtype=torch.float)
            scores_t = torch.tensor(scores, dtype=torch.float)

            # Extract embeddings only for high-confidence detections
            reid_min_conf = embed_cfg.get("min_confidence", 0.6)
            high_conf_mask = scores_t > reid_min_conf
            embeddings = torch.zeros(len(boxes_t), extractor.embedding_dim, device=extractor.device)

            if high_conf_mask.any():
                high_conf_embeds = extractor.extract_from_frame(frame_t, boxes_t[high_conf_mask])
                embeddings[high_conf_mask] = high_conf_embeds

            # Track
            track_ids = tracker.update(boxes_t, scores_t, embeddings, frame_id)

            # Update metrics
            current_ids = set(track_ids[track_ids >= 0].tolist())
            new_ids = current_ids - prev_ids
            metrics.total_tracks = max(metrics.total_tracks, tracker.next_id)

            # Simple ID switch counting (real impl would compare GT)
            if len(prev_ids) > 0 and len(new_ids) > 0:
                metrics.id_switches += len(new_ids)

            prev_ids = current_ids
            metrics.total_detections += len(boxes)

            # Visualization
            if visualize or writer:
                draw_tracks(frame, boxes, track_ids.cpu().numpy(), tracker.hat_active)
        else:
            track_ids = []

        t1 = time.perf_counter()
        frame_time_ms = (t1 - t0) * 1000
        metrics.total_time_ms += frame_time_ms
        metrics.frame_times.append(frame_time_ms)
        metrics.total_frames += 1

        if tracker.hat_active:
            metrics.hat_activations += 1

        # Display
        if visualize:
            # Add FPS overlay
            cv2.putText(
                frame, f"FPS: {1000/max(frame_time_ms, 1):.1f}", (width - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )
            cv2.imshow("HAT Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if writer:
            writer.write(frame)

        # Progress
        if frame_id % 100 == 0:
            print(f"Frame {frame_id}/{total_video_frames or '?'} - "
                  f"Tracks: {tracker.num_tracks}, HAT: {tracker.hat_active}")

        frame_id += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="HAT-ReID Tracker Evaluation")
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Config YAML path")
    parser.add_argument("--no-hat", action="store_true",
                        help="Disable HAT transformation")
    parser.add_argument("--device", default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--visualize", action="store_true",
                        help="Show live visualization")
    parser.add_argument("--output", help="Output video path")
    parser.add_argument("--max-frames", type=int,
                        help="Limit frames to process")

    args = parser.parse_args()

    # Handle output path - if directory, append video filename
    output_path = args.output
    if output_path:
        output_p = Path(output_path)
        if output_p.is_dir() or not output_p.suffix:
            output_p.mkdir(parents=True, exist_ok=True)
            video_name = Path(args.video).stem
            output_path = str(output_p / f"{video_name}_tracked.mp4")

    metrics = run_evaluation(
        args.video,
        config_path=args.config,
        use_hat=not args.no_hat,
        device=args.device,
        visualize=args.visualize,
        output_path=output_path,
        max_frames=args.max_frames,
    )

    print(metrics.summary())


if __name__ == "__main__":
    main()
