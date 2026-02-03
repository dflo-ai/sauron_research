"""Colorblind-safe color palette and ID-to-color mapping for ReID visualization."""

# Okabe-Ito colorblind-safe palette (extended to 20 colors)
# Original 8 colors from Okabe & Ito (2008) + 12 additional distinguishable colors
# All colors in BGR format for OpenCV
OKABE_ITO_PALETTE: list[tuple[int, int, int]] = [
    (86, 180, 233),   # Sky blue
    (0, 158, 115),    # Bluish green
    (240, 228, 66),   # Yellow
    (0, 114, 178),    # Blue
    (213, 94, 0),     # Vermillion
    (204, 121, 167),  # Reddish purple
    (230, 159, 0),    # Orange
    (0, 0, 0),        # Black
    # Extended colors (still colorblind-friendly)
    (153, 153, 153),  # Gray
    (255, 127, 80),   # Coral (BGR: 80, 127, 255)
    (144, 238, 144),  # Light green
    (255, 182, 193),  # Light pink
    (64, 224, 208),   # Turquoise
    (255, 215, 0),    # Gold
    (186, 85, 211),   # Medium orchid
    (0, 206, 209),    # Dark turquoise
    (255, 99, 71),    # Tomato
    (60, 179, 113),   # Medium sea green
    (238, 130, 238),  # Violet
    (127, 255, 212),  # Aquamarine
]


def get_id_color(track_id: int) -> tuple[int, int, int]:
    """Get deterministic BGR color for a track ID.

    Uses modulo indexing into the Okabe-Ito palette to ensure:
    - Same ID always gets same color across all frames
    - Colors are colorblind-accessible
    - O(1) lookup time

    Args:
        track_id: Person track identifier

    Returns:
        BGR color tuple for OpenCV
    """
    if track_id < 0:
        return (128, 128, 128)  # Gray for invalid IDs
    return OKABE_ITO_PALETTE[track_id % len(OKABE_ITO_PALETTE)]


def get_bbox_style(
    confidence: float,
    is_matched: bool,
    thickness_matched: int = 3,
    thickness_unmatched: int = 2,
) -> dict:
    """Get bounding box style based on confidence and match status.

    Args:
        confidence: Detection confidence (0.0-1.0)
        is_matched: Whether this detection was matched via ReID
        thickness_matched: Line thickness for matched detections
        thickness_unmatched: Line thickness for unmatched detections

    Returns:
        Dict with 'thickness' and 'opacity' keys
    """
    # Base thickness from match status
    thickness = thickness_matched if is_matched else thickness_unmatched

    # Confidence affects opacity (0.5-1.0 range)
    opacity = 0.5 + (confidence * 0.5)

    return {"thickness": thickness, "opacity": opacity}
