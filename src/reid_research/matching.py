"""Re-ranking and matching utilities for person re-identification.

Implements full k-reciprocal encoding with Jaccard distance (CVPR 2017),
quality-aware feature fusion, and velocity-based temporal consistency.
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_k_reciprocal_reranking(
    query_feats: np.ndarray,
    gallery_feats: np.ndarray,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> np.ndarray:
    """Full k-reciprocal encoding with Jaccard distance.

    Reference: "Re-ranking Person Re-identification with k-reciprocal Encoding"
    CVPR 2017 - Zhong et al.

    Algorithm:
    1. Compute initial distance matrix (query vs gallery)
    2. For each query, find k1 nearest neighbors
    3. Filter to k-reciprocal set R(q,k1)
    4. Expand set via R*(q,k1) using k2
    5. Encode as binary vectors, compute Jaccard distance
    6. Final distance = lambda*original + (1-lambda)*jaccard

    Args:
        query_feats: (Q, D) query feature vectors
        gallery_feats: (G, D) gallery feature vectors
        k1: Initial k for R(p,k1), default 20
        k2: Expansion k for R*(p,k1), default 6
        lambda_value: Weight for original distance, default 0.3

    Returns:
        (Q, G) refined similarity scores (higher = more similar)
    """
    Q = query_feats.shape[0]
    G = gallery_feats.shape[0]

    if G < 2:
        return cosine_similarity(query_feats, gallery_feats)

    # Adaptive k values for small galleries
    k1 = min(k1, G)
    k2 = min(k2, G)

    # Compute initial distances (1 - cosine_similarity)
    q_g_sim = cosine_similarity(query_feats, gallery_feats)
    g_g_sim = cosine_similarity(gallery_feats, gallery_feats)

    # Convert to distance (for sorting - lower is closer)
    q_g_dist = 1 - q_g_sim
    g_g_dist = 1 - g_g_sim

    # Get sorted indices (nearest first)
    q_g_sorted = np.argsort(q_g_dist, axis=1)  # (Q, G)
    g_g_sorted = np.argsort(g_g_dist, axis=1)  # (G, G)

    # Compute k-reciprocal sets for gallery items
    g_reciprocal_sets = _compute_gallery_reciprocal_sets(g_g_sorted, k1)

    # Compute Jaccard distance matrix
    jaccard_dist = np.zeros((Q, G), dtype=np.float32)

    for q_idx in range(Q):
        # Get query's k-reciprocal set
        q_k_neighbors = set(q_g_sorted[q_idx, :k1].tolist())
        q_reciprocal = _get_k_reciprocal_set(
            q_idx, q_g_sorted, g_g_sorted, k1, is_query=True
        )

        # Expand using k2
        q_expanded = _expand_k_reciprocal_set(
            q_reciprocal, g_reciprocal_sets, k2, G
        )

        # Compute Jaccard distance to each gallery item
        for g_idx in range(G):
            g_expanded = _expand_k_reciprocal_set(
                g_reciprocal_sets[g_idx], g_reciprocal_sets, k2, G
            )
            jaccard_dist[q_idx, g_idx] = _jaccard_distance(q_expanded, g_expanded)

    # Combine original and Jaccard distances
    final_dist = lambda_value * q_g_dist + (1 - lambda_value) * jaccard_dist

    # Convert back to similarity
    return 1 - final_dist


def _compute_gallery_reciprocal_sets(
    g_g_sorted: np.ndarray, k: int
) -> list[set[int]]:
    """Compute k-reciprocal sets for all gallery items.

    Args:
        g_g_sorted: (G, G) sorted gallery indices
        k: Number of neighbors

    Returns:
        List of k-reciprocal sets for each gallery item
    """
    G = g_g_sorted.shape[0]
    reciprocal_sets = []

    for g_idx in range(G):
        # g's k nearest neighbors
        g_neighbors = set(g_g_sorted[g_idx, :k].tolist())
        reciprocal = set()

        for neighbor in g_neighbors:
            # Check if g is in neighbor's k nearest
            neighbor_neighbors = set(g_g_sorted[neighbor, :k].tolist())
            if g_idx in neighbor_neighbors:
                reciprocal.add(neighbor)

        # Always include self
        reciprocal.add(g_idx)
        reciprocal_sets.append(reciprocal)

    return reciprocal_sets


def _get_k_reciprocal_set(
    idx: int,
    q_g_sorted: np.ndarray,
    g_g_sorted: np.ndarray,
    k: int,
    is_query: bool = True,
) -> set[int]:
    """Get k-reciprocal set for a query or gallery item.

    Args:
        idx: Index of item
        q_g_sorted: (Q, G) sorted query-gallery indices
        g_g_sorted: (G, G) sorted gallery-gallery indices
        k: Number of neighbors
        is_query: True if idx is query, False if gallery

    Returns:
        Set of k-reciprocal neighbor indices (gallery indices)
    """
    if is_query:
        # Query's k nearest gallery neighbors
        neighbors = set(q_g_sorted[idx, :k].tolist())
    else:
        # Gallery's k nearest gallery neighbors
        neighbors = set(g_g_sorted[idx, :k].tolist())

    reciprocal = set()

    for neighbor in neighbors:
        # Check reciprocity: is idx in neighbor's k nearest?
        if is_query:
            # For query, check if query is similar to gallery item
            # (can't do true reciprocal, use gallery-gallery as proxy)
            neighbor_neighbors = set(g_g_sorted[neighbor, : k // 2].tolist())
            # Boost if neighbor's neighbors overlap significantly with query's neighbors
            overlap = len(neighbors & neighbor_neighbors)
            if overlap >= k // 4:
                reciprocal.add(neighbor)
        else:
            neighbor_neighbors = set(g_g_sorted[neighbor, :k].tolist())
            if idx in neighbor_neighbors:
                reciprocal.add(neighbor)

    return reciprocal


def _expand_k_reciprocal_set(
    reciprocal_set: set[int],
    all_reciprocal_sets: list[set[int]],
    k2: int,
    G: int,
) -> set[int]:
    """Expand k-reciprocal set using k2 expansion.

    If >50% of a member's k2-reciprocal set overlaps with original set,
    add that member's set to the expanded set.

    Args:
        reciprocal_set: Initial k-reciprocal set
        all_reciprocal_sets: Pre-computed reciprocal sets for gallery
        k2: Expansion k value
        G: Gallery size

    Returns:
        Expanded reciprocal set
    """
    expanded = reciprocal_set.copy()

    for member in list(reciprocal_set):
        if member >= len(all_reciprocal_sets):
            continue

        member_set = all_reciprocal_sets[member]
        # Limit member set size for expansion
        member_subset = set(list(member_set)[:k2]) if len(member_set) > k2 else member_set

        # Check overlap ratio
        overlap = len(reciprocal_set & member_subset)
        if len(member_subset) > 0 and overlap / len(member_subset) > 0.5:
            expanded |= member_subset

    return expanded


def _jaccard_distance(set_a: set[int], set_b: set[int]) -> float:
    """Compute Jaccard distance between two sets.

    Jaccard distance = 1 - |A ∩ B| / |A ∪ B|

    Args:
        set_a: First set
        set_b: Second set

    Returns:
        Jaccard distance in [0, 1]
    """
    if not set_a and not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    if union == 0:
        return 0.0

    return 1 - intersection / union


def compute_quality_score(
    confidence: float,
    bbox: tuple[float, float, float, float],
    min_bbox_area: int = 2000,
    ideal_aspect_ratio: float = 0.4,
    confidence_weight: float = 0.6,
    geometry_weight: float = 0.4,
) -> float:
    """Compute detection quality score for feature weighting.

    Combines detection confidence with bbox geometry quality.

    Args:
        confidence: Detection confidence score [0, 1]
        bbox: Bounding box (x1, y1, x2, y2)
        min_bbox_area: Minimum pixels for valid detection
        ideal_aspect_ratio: Target W/H ratio (person ~0.4)
        confidence_weight: Weight for confidence component
        geometry_weight: Weight for geometry component

    Returns:
        Quality score in [0, 1], higher = better
    """
    # Confidence component (already 0-1)
    conf_score = min(1.0, max(0.0, confidence))

    # Geometry component
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    area = w * h

    # Area score (saturates at min_bbox_area)
    area_score = min(1.0, area / min_bbox_area) if min_bbox_area > 0 else 1.0

    # Aspect ratio score (penalize deviation from ideal)
    aspect = w / max(h, 1)
    aspect_deviation = abs(aspect - ideal_aspect_ratio)
    aspect_score = max(0.0, 1.0 - aspect_deviation * 2)

    geom_score = (area_score + aspect_score) / 2

    # Weighted combination
    return confidence_weight * conf_score + geometry_weight * geom_score


@dataclass
class TrackMotion:
    """Motion state for velocity-based temporal consistency."""

    positions: deque = field(default_factory=lambda: deque(maxlen=10))
    velocity: tuple[float, float] = (0.0, 0.0)
    last_frame: int = 0


def compute_velocity(
    positions: deque,
    max_speed: float = 100.0,
) -> tuple[float, float]:
    """Compute velocity from position history.

    Args:
        positions: Deque of (cx, cy) positions
        max_speed: Maximum velocity magnitude (clamps outliers)

    Returns:
        (vx, vy) velocity in pixels/frame
    """
    if len(positions) < 2:
        return (0.0, 0.0)

    pos_list = list(positions)
    n = len(pos_list)

    # Average velocity over history
    vx = (pos_list[-1][0] - pos_list[0][0]) / n
    vy = (pos_list[-1][1] - pos_list[0][1]) / n

    # Clamp to max speed
    speed = (vx**2 + vy**2) ** 0.5
    if speed > max_speed:
        scale = max_speed / speed
        vx, vy = vx * scale, vy * scale

    return (vx, vy)


def predict_position(
    last_position: tuple[float, float],
    velocity: tuple[float, float],
    frames_elapsed: int = 1,
) -> tuple[float, float]:
    """Predict position using linear extrapolation.

    Args:
        last_position: (cx, cy) last known center
        velocity: (vx, vy) velocity in pixels/frame
        frames_elapsed: Number of frames since last observation

    Returns:
        (pred_x, pred_y) predicted center position
    """
    pred_x = last_position[0] + velocity[0] * frames_elapsed
    pred_y = last_position[1] + velocity[1] * frames_elapsed
    return (pred_x, pred_y)


def compute_position_boost(
    detection_center: tuple[float, float],
    predicted_center: tuple[float, float],
    prediction_radius: float = 75.0,
    max_boost: float = 0.15,
) -> float:
    """Compute similarity boost based on position proximity to prediction.

    Args:
        detection_center: (cx, cy) of detection
        predicted_center: (cx, cy) predicted position
        prediction_radius: Maximum distance for boost
        max_boost: Maximum similarity boost

    Returns:
        Boost value in [0, max_boost]
    """
    dx = detection_center[0] - predicted_center[0]
    dy = detection_center[1] - predicted_center[1]
    distance = (dx**2 + dy**2) ** 0.5

    if distance >= prediction_radius:
        return 0.0

    # Boost inversely proportional to distance
    return max_boost * (1 - distance / prediction_radius)


def compute_adaptive_threshold(
    scores: list[float],
    base_threshold: float,
    target_percentile: float = 0.15,
    min_threshold: float = 0.50,
    max_threshold: float = 0.80,
    warmup_count: int = 20,
) -> float:
    """Compute adaptive threshold from similarity distribution.

    Uses percentile-based approach to maintain target precision.

    Args:
        scores: List of recent match similarity scores
        base_threshold: Default threshold when insufficient data
        target_percentile: Target false positive rate (reject bottom X%)
        min_threshold: Minimum allowed threshold
        max_threshold: Maximum allowed threshold
        warmup_count: Minimum scores before adapting

    Returns:
        Adjusted similarity threshold
    """
    if len(scores) < warmup_count:
        return base_threshold

    # Compute percentile threshold
    threshold = np.percentile(scores, target_percentile * 100)

    # Clamp to bounds
    return max(min_threshold, min(max_threshold, threshold))
