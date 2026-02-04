"""Configuration models and YAML loader for ReID research module."""
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model configuration."""

    reid_variant: str = "osnet_x1_0"
    reid_weights: str | None = None
    device: str = "cuda"
    # JointBDOE detector weights
    yolo_weights: str = "data/weights/jointbdoe_m.pt"


class InferenceConfig(BaseModel):
    """Inference parameters."""

    image_size: tuple[int, int] = (256, 128)
    batch_size: int = 32
    confidence_threshold: float = 0.75
    similarity_threshold: float = 0.5


class GalleryConfig(BaseModel):
    """Gallery management parameters."""

    max_features_per_id: int = 10
    ema_alpha: float = 0.7

    # Rank-list majority voting (triplet-loss style)
    rank_list_size: int = 20  # Top-k entries in rank list (max 50)
    rank_distance_threshold: float | None = None  # Auto (median) if None
    rank_min_entries_per_id: int = 3  # Min entries before voting kicks in
    rank_fallback_threshold: float = 1.2  # Distance threshold for early matching

    # Full k-reciprocal re-ranking with Jaccard distance (CVPR 2017)
    use_full_reranking: bool = False  # +5-15% mAP, slower
    rerank_k1: int = 20  # Initial k for R(p,k1)
    rerank_k2: int = 6  # Expansion k for R*(p,k1)
    rerank_lambda: float = 0.3  # Weight for original distance

    # Quality-aware feature fusion
    use_quality_weighting: bool = True
    quality_min_threshold: float = 0.3  # Skip updates below this
    quality_confidence_weight: float = 0.6  # Weight for detection confidence
    quality_geometry_weight: float = 0.4  # Weight for bbox geometry
    ideal_aspect_ratio: float = 0.4  # Typical person W/H ratio
    min_bbox_area: int = 2000  # Minimum pixels for valid detection

    # Velocity-based temporal consistency (replaces position-hash)
    use_velocity_prediction: bool = True
    velocity_history_frames: int = 5  # Frames to average velocity
    velocity_max_speed: float = 100.0  # Max pixels/frame (clamp outliers)
    prediction_radius: float = 75.0  # Match radius around prediction
    prediction_boost: float = 0.15  # Similarity boost for position match

    # Adaptive similarity threshold
    use_adaptive_threshold: bool = False
    adaptive_min_threshold: float = 0.50  # Floor
    adaptive_max_threshold: float = 0.80  # Ceiling
    adaptive_window_size: int = 100  # Matches to track
    adaptive_target_percentile: float = 0.15  # Target false positive rate
    adaptive_warmup_matches: int = 20  # Minimum matches before adapting

    # Crossing detection (Phase 2)
    use_crossing_detection: bool = True  # Enable crossing detection
    crossing_detection_radius: float = 100.0  # Distance threshold for crossing (pixels)
    crossing_iou_threshold: float = 0.1  # IoU threshold for bbox overlap detection
    crossing_threshold_boost: float = 0.6  # Stricter similarity threshold during crossing
    crossing_boost_reduction: float = 0.3  # Reduce position boost to 30% during crossing
    rerank_on_crossing: bool = True  # Apply full torchreid re-ranking during crossing

    # Adaptive cost matrix (Phase 3)
    use_adaptive_cost_matrix: bool = False  # Use adaptive cost matrix vs simple 1-sim
    weight_appearance: float = 0.7  # Appearance weight (normal)
    weight_motion: float = 0.3  # Motion weight (normal)
    weight_appearance_crossing: float = 0.9  # Appearance weight (crossing)
    weight_motion_crossing: float = 0.1  # Motion weight (crossing)

    # Motion consistency validation (Phase 5)
    use_motion_validation: bool = True  # Validate assignments against motion history
    motion_max_distance: float = 150.0  # Max distance from prediction (pixels)
    motion_direction_threshold: float = 120.0  # Max angle change (degrees)

    # Tentative track confirmation (require N frames before permanent ID)
    min_frames_for_id: int = 5  # Min consecutive frames before assigning permanent ID
    tentative_max_age: int = 10  # Max frames to keep unconfirmed tentative track


class VisualizationConfig(BaseModel):
    """Visualization configuration for video output."""

    # Layout
    gallery_panel_width: int = 200
    gallery_panel_position: str = "right"  # "left" or "right"
    show_gallery_panel: bool = True
    show_pipeline_hud: bool = True
    max_gallery_entries: int = 10  # Most recent entries to show

    # Colors and styling
    panel_bg_opacity: float = 0.75
    bbox_thickness_matched: int = 3
    bbox_thickness_unmatched: int = 2

    # Extended frame layout (analytics outside video frame)
    extended_frame_enabled: bool = True  # Default: use extended layout


class OutputConfig(BaseModel):
    """Output configuration."""

    save_video: bool = True
    save_tracks: bool = True
    visualization: bool = True


class DebugConfig(BaseModel):
    """Debug and diagnostic configuration."""

    # ID switch frame capture (for investigating Re-ID failures)
    capture_id_switches: bool = False  # Enable ID switch frame capture
    id_switch_output_dir: str = "outputs/id_switches"  # Base output directory
    id_switch_frames_before: int = 3  # Frames to capture before switch
    id_switch_frames_after: int = 3  # Frames to capture after switch


class ReIDConfig(BaseModel):
    """Root configuration model."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    gallery: GalleryConfig = Field(default_factory=GalleryConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)


def load_config(path: str | Path) -> ReIDConfig:
    """Load config from YAML file.

    Args:
        path: Path to YAML config file

    Returns:
        Validated ReIDConfig instance
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    return ReIDConfig(**data)
