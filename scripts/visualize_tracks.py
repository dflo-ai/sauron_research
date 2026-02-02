#!/usr/bin/env python3
"""Visualize tracking results with debug info.

Usage:
    python scripts/visualize_tracks.py data/videos/test.mp4
    python scripts/visualize_tracks.py data/videos/test.mp4 --save-frames output/
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_color_palette(n_colors: int = 100) -> list[tuple[int, int, int]]:
    """Generate distinct colors for track visualization."""
    colors = []
    for i in range(n_colors):
        hue = int(180 * i / n_colors)
        hsv = np.uint8([[[hue, 255, 200]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, bgr)))
    return colors


def draw_track_info(
    frame: np.ndarray,
    track_id: int,
    bbox: tuple[int, int, int, int],
    score: float,
    color: tuple[int, int, int],
    show_score: bool = True,
) -> None:
    """Draw single track with bounding box and label."""
    x1, y1, x2, y2 = bbox

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Draw label background
    label = f"ID:{track_id}"
    if show_score:
        label += f" {score:.2f}"

    (label_w, label_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )

    cv2.rectangle(
        frame,
        (x1, y1 - label_h - baseline - 5),
        (x1 + label_w + 5, y1),
        color,
        -1
    )

    # Draw label text
    cv2.putText(
        frame, label, (x1 + 2, y1 - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )


def draw_debug_overlay(
    frame: np.ndarray,
    frame_id: int,
    num_tracks: int,
    hat_active: bool,
    fps: float,
) -> None:
    """Draw debug information overlay."""
    h, w = frame.shape[:2]

    # Background panel
    cv2.rectangle(frame, (5, 5), (200, 100), (0, 0, 0), -1)
    cv2.rectangle(frame, (5, 5), (200, 100), (255, 255, 255), 1)

    # Debug text
    lines = [
        f"Frame: {frame_id}",
        f"Tracks: {num_tracks}",
        f"HAT: {'ON' if hat_active else 'OFF'}",
        f"FPS: {fps:.1f}",
    ]

    y = 25
    for line in lines:
        color = (0, 255, 0) if "ON" in line else (255, 255, 255)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 20


def visualize_video(
    video_path: str,
    output_dir: str | None = None,
    show_preview: bool = True,
    max_frames: int | None = None,
) -> None:
    """Visualize video with simulated tracks (for testing visualization).

    In real usage, tracks would come from the tracker.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height} @ {fps:.1f} FPS")

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving frames to: {output_path}")

    colors = get_color_palette(50)
    frame_id = 0

    # Simulated tracks for demo (in real usage, these come from tracker)
    demo_tracks = [
        {"id": 0, "x": 100, "y": 100, "vx": 2, "vy": 1},
        {"id": 1, "x": 300, "y": 200, "vx": -1, "vy": 2},
        {"id": 2, "x": 500, "y": 150, "vx": 1, "vy": -1},
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames and frame_id >= max_frames:
            break

        # Update and draw simulated tracks
        for track in demo_tracks:
            # Simple motion simulation
            track["x"] += track["vx"]
            track["y"] += track["vy"]

            # Bounce off edges
            if track["x"] < 50 or track["x"] > width - 150:
                track["vx"] *= -1
            if track["y"] < 50 or track["y"] > height - 200:
                track["vy"] *= -1

            # Draw track
            x, y = int(track["x"]), int(track["y"])
            bbox = (x, y, x + 100, y + 150)
            color = colors[track["id"] % len(colors)]
            draw_track_info(frame, track["id"], bbox, 0.9, color)

        # Draw debug overlay
        hat_active = frame_id > 100  # Simulate HAT activation
        draw_debug_overlay(frame, frame_id, len(demo_tracks), hat_active, fps)

        # Save frame
        if output_dir:
            cv2.imwrite(str(output_path / f"frame_{frame_id:05d}.jpg"), frame)

        # Show preview
        if show_preview:
            cv2.imshow("Track Visualization", frame)
            key = cv2.waitKey(int(1000 / fps))
            if key == ord("q"):
                break
            elif key == ord(" "):  # Pause
                cv2.waitKey(0)

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_id} frames")


def main():
    parser = argparse.ArgumentParser(description="Track Visualization Tool")
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--save-frames", dest="output_dir",
                        help="Save frames to directory")
    parser.add_argument("--no-preview", action="store_true",
                        help="Disable live preview")
    parser.add_argument("--max-frames", type=int,
                        help="Limit frames to process")

    args = parser.parse_args()

    visualize_video(
        args.video,
        output_dir=args.output_dir,
        show_preview=not args.no_preview,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
