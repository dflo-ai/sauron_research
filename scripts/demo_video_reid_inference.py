#!/usr/bin/env python3
"""Demo script for video ReID inference.

Usage:
    python scripts/demo_video_reid_inference.py --video input.mp4 --output outputs/
    python scripts/demo_video_reid_inference.py --video input.mp4 --config configs/default.yaml
"""
import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reid_research import ReIDConfig, VideoReIDPipeline, load_config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Video ReID inference demo")
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        default=None,
        help="Input video path (default: first video in data/videos/)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="outputs/",
        help="Output directory (default: outputs/)",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/default.yaml",
        help="Config file path (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda/cpu)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override similarity threshold",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip video output (tracks only)",
    )
    # Visualization options
    parser.add_argument(
        "--no-gallery",
        action="store_true",
        help="Disable gallery side panel",
    )
    parser.add_argument(
        "--no-hud",
        action="store_true",
        help="Disable pipeline HUD",
    )
    parser.add_argument(
        "--split-view",
        action="store_true",
        help="Enable split-view layout (side-by-side pipeline stages)",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        print(f"Loading config: {config_path}")
        config = load_config(config_path)
    else:
        print("Using default config")
        config = ReIDConfig()

    # Apply CLI overrides
    if args.device:
        config.model.device = args.device
    if args.threshold:
        config.inference.similarity_threshold = args.threshold
    if args.no_video:
        config.output.save_video = False

    # Visualization overrides
    if args.no_gallery:
        config.visualization.show_gallery_panel = False
    if args.no_hud:
        config.visualization.show_pipeline_hud = False
    if args.split_view:
        config.visualization.split_view_enabled = True

    # Setup paths (validated: use data/videos/ by default)
    if args.video:
        video_path = Path(args.video)
    else:
        # Auto-find first video in data/videos/
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

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_video = output_dir / f"{video_path.stem}_tracked.mp4"
    output_tracks = output_dir / f"{video_path.stem}_tracks.json"
    output_gallery = output_dir / f"{video_path.stem}_gallery.json"

    # Create pipeline
    print("Initializing pipeline...")
    pipeline = VideoReIDPipeline(config)

    # Process video
    print(f"Processing: {video_path}")
    stats = pipeline.process_video(
        video_path=video_path,
        output_path=output_video if config.output.save_video else None,
    )

    # Export results
    if config.output.save_tracks:
        gallery_features = pipeline.gallery.export_features()
        tracks_data = {
            "video": str(video_path),
            "config": {
                "reid_variant": config.model.reid_variant,
                "similarity_threshold": config.inference.similarity_threshold,
            },
            "stats": {
                "frames": stats["frames"],
                "detections": stats["detections"],
                "unique_persons": stats["unique_ids"],
            },
            "persons": {
                str(tid): {
                    "feature_dim": len(feat),
                }
                for tid, feat in gallery_features.items()
            },
        }
        with open(output_tracks, "w") as f:
            json.dump(tracks_data, f, indent=2)
        print(f"Tracks saved: {output_tracks}")

        # Save gallery for later reuse
        pipeline.gallery.save(output_gallery)
        print(f"Gallery saved: {output_gallery}")

    # Print summary
    print("\n=== Results ===")
    print(f"Frames processed: {stats['frames']}")
    print(f"Total detections: {stats['detections']}")
    print(f"Unique persons: {stats['unique_ids']}")
    if config.output.save_video:
        print(f"Output video: {output_video}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
