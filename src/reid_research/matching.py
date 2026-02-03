"""Re-ranking and matching utilities for person re-identification.

Implements full k-reciprocal encoding with Jaccard distance (CVPR 2017),
quality-aware feature fusion, and velocity-based temporal consistency.
Includes official torchreid re-ranking implementation.
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# Official Torchreid Re-ranking (CVPR 2017)
# Source: deep-person-reid/torchreid/utils/rerank.py
# Paper: "Re-ranking Person Re-identification with k-reciprocal Encoding"
# =============================================================================


def euclidean_squared_distance_np(input1: np.ndarray, input2: np.ndarray) -> np.ndarray:
    """Compute euclidean squared distance matrix (numpy version).

    Efficient implementation using: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b

    Args:
        input1: (N, D) feature matrix
        input2: (M, D) feature matrix

    Returns:
        (N, M) distance matrix where dist[i,j] = ||input1[i] - input2[j]||^2
    """
    mat1 = np.sum(input1**2, axis=1, keepdims=True)  # (N, 1)
    mat2 = np.sum(input2**2, axis=1, keepdims=True).T  # (1, M)
    distmat = mat1 + mat2 - 2 * np.dot(input1, input2.T)
    return np.maximum(distmat, 0)  # Numerical stability


def re_ranking_torchreid(
    q_g_dist: np.ndarray,
    q_q_dist: np.ndarray,
    g_g_dist: np.ndarray,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> np.ndarray:
    """Official CVPR2017 k-reciprocal re-ranking from torchreid.

    Source: deep-person-reid/torchreid/utils/rerank.py
    Paper: Zhong et al., "Re-ranking Person Re-identification with
           k-reciprocal Encoding", CVPR 2017.

    Args:
        q_g_dist: (Q, G) query-gallery distance matrix
        q_q_dist: (Q, Q) query-query distance matrix
        g_g_dist: (G, G) gallery-gallery distance matrix
        k1: Initial k for R(p,k1), default 20
        k2: Expansion k for query expansion, default 6
        lambda_value: Weight for original distance, default 0.3

    Returns:
        (Q, G) re-ranked distance matrix
    """
    # Concatenate all distances into unified matrix
    original_dist = np.concatenate(
        [
            np.concatenate([q_q_dist, q_g_dist], axis=1),
            np.concatenate([q_g_dist.T, g_g_dist], axis=1),
        ],
        axis=0,
    )
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1.0 * original_dist / np.max(original_dist, axis=0))

    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    # Build k-reciprocal encoding vectors
    for i in range(all_num):
        forward_k_neigh_index = initial_rank[i, : k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, : k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index

        # Expand k-reciprocal set
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[
                candidate, : int(np.around(k1 / 2.0)) + 1
            ]
            candidate_backward_k_neigh_index = initial_rank[
                candidate_forward_k_neigh_index, : int(np.around(k1 / 2.0)) + 1
            ]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if (
                len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index))
                > 2.0 / 3 * len(candidate_k_reciprocal_index)
            ):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index
                )

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = 1.0 * weight / np.sum(weight)

    original_dist = original_dist[:query_num,]

    # Query expansion via averaging V vectors
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe

    del initial_rank

    # Build inverted index for Jaccard computation
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    # Compute Jaccard distance
    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]]
            )
        jaccard_dist[i] = 1 - temp_min / (2.0 - temp_min)

    # Final distance: weighted combination
    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


def apply_torchreid_reranking(
    query_feats: np.ndarray,
    gallery_feats: np.ndarray,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> np.ndarray:
    """Apply official torchreid re-ranking and return similarity matrix.

    This is a convenience wrapper that:
    1. Computes the 3 required distance matrices (q_q, q_g, g_g)
    2. Applies the official CVPR2017 re-ranking
    3. Converts final distances back to similarities

    Args:
        query_feats: (Q, D) query feature vectors (L2 normalized)
        gallery_feats: (G, D) gallery feature vectors (L2 normalized)
        k1: Initial k for R(p,k1), default 20
        k2: Expansion k for query expansion, default 6
        lambda_value: Weight for original distance, default 0.3

    Returns:
        (Q, G) similarity scores (higher = more similar)
    """
    # Compute distance matrices
    q_g_dist = euclidean_squared_distance_np(query_feats, gallery_feats)
    q_q_dist = euclidean_squared_distance_np(query_feats, query_feats)
    g_g_dist = euclidean_squared_distance_np(gallery_feats, gallery_feats)

    # Apply re-ranking
    final_dist = re_ranking_torchreid(
        q_g_dist, q_q_dist, g_g_dist, k1, k2, lambda_value
    )

    # Convert distance to similarity: sim = 1 / (1 + dist)
    # This maps [0, inf) -> (0, 1]
    similarity = 1.0 / (1.0 + final_dist)
    return similarity


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


# =============================================================================
# Crossing Detection Module (Phase 2)
# Detects when tracks are crossing or about to cross
# =============================================================================


def compute_iou(
    bbox1: tuple[float, float, float, float],
    bbox2: tuple[float, float, float, float],
) -> float:
    """Compute Intersection over Union between two bounding boxes.

    Args:
        bbox1: (x1, y1, x2, y2) first bounding box
        bbox2: (x1, y1, x2, y2) second bounding box

    Returns:
        IoU value in [0, 1]
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def velocities_converging(
    pos1: tuple[float, float],
    vel1: tuple[float, float],
    pos2: tuple[float, float],
    vel2: tuple[float, float],
) -> bool:
    """Check if two objects with given velocities are converging.

    Two objects are converging if at least one is moving toward the other.

    Args:
        pos1: (cx, cy) position of first object
        vel1: (vx, vy) velocity of first object
        pos2: (cx, cy) position of second object
        vel2: (vx, vy) velocity of second object

    Returns:
        True if objects are converging
    """
    # Vector from pos1 to pos2
    dx, dy = pos2[0] - pos1[0], pos2[1] - pos1[1]

    # Dot product of vel1 with direction to pos2 (positive = moving toward)
    dot1 = vel1[0] * dx + vel1[1] * dy

    # Dot product of vel2 with direction to pos1 (positive = moving toward)
    dot2 = vel2[0] * (-dx) + vel2[1] * (-dy)

    # At least one moving toward the other
    return dot1 > 0 or dot2 > 0


def detect_crossing_tracks(
    track_positions: dict[int, tuple[float, float]],
    track_velocities: dict[int, tuple[float, float]],
    track_bboxes: dict[int, tuple[float, float, float, float]] | None = None,
    crossing_radius: float = 100.0,
    iou_threshold: float = 0.1,
) -> set[int]:
    """Detect tracks that are crossing or about to cross.

    A crossing is detected when:
    1. Two tracks are within crossing_radius AND velocities converging, OR
    2. Two tracks have bbox IoU > iou_threshold

    Args:
        track_positions: Dict of track_id -> (cx, cy) current/predicted position
        track_velocities: Dict of track_id -> (vx, vy) velocity
        track_bboxes: Optional dict of track_id -> (x1, y1, x2, y2) bounding boxes
        crossing_radius: Distance threshold for crossing detection (pixels)
        iou_threshold: IoU threshold for bbox overlap detection

    Returns:
        Set of track_ids involved in crossing
    """
    crossing_ids: set[int] = set()
    track_ids = list(track_positions.keys())

    for i, tid1 in enumerate(track_ids):
        pos1 = track_positions[tid1]
        vel1 = track_velocities.get(tid1, (0.0, 0.0))

        for tid2 in track_ids[i + 1 :]:
            pos2 = track_positions[tid2]
            vel2 = track_velocities.get(tid2, (0.0, 0.0))

            # Check distance
            dist = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

            # Method 1: Distance + velocity convergence
            velocity_crossing = False
            if dist < crossing_radius:
                velocity_crossing = velocities_converging(pos1, vel1, pos2, vel2)

            # Method 2: Bbox IoU overlap (more robust for close encounters)
            iou_crossing = False
            if track_bboxes and tid1 in track_bboxes and tid2 in track_bboxes:
                iou = compute_iou(track_bboxes[tid1], track_bboxes[tid2])
                iou_crossing = iou > iou_threshold

            if velocity_crossing or iou_crossing:
                crossing_ids.add(tid1)
                crossing_ids.add(tid2)

    return crossing_ids


def validate_motion_consistency(
    detection_center: tuple[float, float],
    predicted_center: tuple[float, float],
    detection_velocity: tuple[float, float],
    track_velocity: tuple[float, float],
    max_distance: float = 150.0,
    direction_threshold_deg: float = 120.0,
) -> bool:
    """Validate if detection motion is consistent with track prediction.

    Used to reject ID theft during crossing by checking:
    1. Detection is near predicted position
    2. Detection's implied velocity direction is consistent with track

    Args:
        detection_center: (cx, cy) of detection
        predicted_center: (cx, cy) predicted track position
        detection_velocity: (vx, vy) implied velocity of detection
        track_velocity: (vx, vy) track's velocity history
        max_distance: Maximum allowed distance from prediction
        direction_threshold_deg: Maximum angle difference (degrees)

    Returns:
        True if motion is consistent (valid match)
    """
    # For stationary/slow-moving tracks, skip validation
    # Motion validation is designed for moving objects - stationary people
    # have natural detection jitter that shouldn't cause rejection
    track_speed = (track_velocity[0] ** 2 + track_velocity[1] ** 2) ** 0.5
    if track_speed < 5.0:  # Near-stationary track
        return True  # Always valid for stationary people

    # Check position proximity
    dx = detection_center[0] - predicted_center[0]
    dy = detection_center[1] - predicted_center[1]
    distance = (dx**2 + dy**2) ** 0.5

    if distance > max_distance:
        return False

    # Check velocity direction consistency
    det_speed = (detection_velocity[0] ** 2 + detection_velocity[1] ** 2) ** 0.5

    if det_speed < 1.0 or track_speed < 1.0:
        # Can't reliably check direction, pass based on position only
        return True

    # Compute angle between velocity vectors
    dot = detection_velocity[0] * track_velocity[0] + detection_velocity[1] * track_velocity[1]
    cos_angle = dot / (det_speed * track_speed)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp for numerical stability

    angle_deg = np.arccos(cos_angle) * 180.0 / np.pi

    return angle_deg <= direction_threshold_deg


# =============================================================================
# Adaptive Cost Matrix (Phase 3)
# Dynamic weighting based on crossing state
# =============================================================================


def compute_adaptive_cost_matrix(
    sim_matrix: np.ndarray,
    detection_positions: list[tuple[float, float]],
    predicted_positions: dict[int, tuple[float, float]],
    gallery_ids: list[int],
    crossing_ids: set[int],
    weights_normal: tuple[float, float] = (0.7, 0.3),
    weights_crossing: tuple[float, float] = (0.9, 0.1),
    max_position_distance: float = 200.0,
) -> np.ndarray:
    """Compute adaptive cost matrix with dynamic weighting.

    Combines appearance and motion costs with different weights based on
    whether tracks are crossing. During crossing, appearance dominates.

    Args:
        sim_matrix: (Q, G) appearance similarity scores
        detection_positions: List of (cx, cy) for each query detection
        predicted_positions: Dict of gallery_id -> predicted (cx, cy)
        gallery_ids: List of gallery IDs corresponding to sim_matrix columns
        crossing_ids: Set of track IDs in crossing state
        weights_normal: (appearance, motion) weights for normal state
        weights_crossing: Weights for crossing state
        max_position_distance: Normalize distance by this value

    Returns:
        (Q, G) cost matrix for Hungarian algorithm (lower = better match)
    """
    Q, G = sim_matrix.shape
    cost_matrix = np.zeros((Q, G), dtype=np.float32)

    # Appearance cost: 1 - similarity (lower sim = higher cost)
    appearance_cost = 1.0 - sim_matrix

    for q_idx in range(Q):
        det_pos = detection_positions[q_idx]

        for g_idx, gal_id in enumerate(gallery_ids):
            # Determine weights based on crossing state
            if gal_id in crossing_ids:
                w_app, w_motion = weights_crossing
            else:
                w_app, w_motion = weights_normal

            # Motion cost: normalized position distance
            motion_cost = 1.0  # Max cost if no prediction
            if gal_id in predicted_positions:
                pred_pos = predicted_positions[gal_id]
                dist = (
                    (det_pos[0] - pred_pos[0]) ** 2 + (det_pos[1] - pred_pos[1]) ** 2
                ) ** 0.5
                motion_cost = min(1.0, dist / max_position_distance)

            # Combined cost (weighted sum)
            cost_matrix[q_idx, g_idx] = (
                w_app * appearance_cost[q_idx, g_idx] + w_motion * motion_cost
            )

    return cost_matrix


def validate_assignments_batch(
    assignments: list[tuple[int | None, float]],
    detection_positions: list[tuple[float, float]],
    predicted_positions: dict[int, tuple[float, float]],
    track_velocities: dict[int, tuple[float, float]],
    crossing_ids: set[int],
    max_distance: float = 150.0,
    direction_threshold_deg: float = 120.0,
    high_similarity_threshold: float = 0.7,
) -> list[tuple[int | None, float]]:
    """Validate batch of assignments against motion consistency.

    Rejects assignments where motion is inconsistent with track history.
    Exemptions: crossing tracks, high-similarity matches, no prediction.

    Args:
        assignments: List of (track_id, similarity) tuples
        detection_positions: (cx, cy) for each detection
        predicted_positions: track_id -> predicted (cx, cy)
        track_velocities: track_id -> (vx, vy)
        crossing_ids: Tracks currently crossing (exempt from validation)
        max_distance: Max allowed distance from prediction
        direction_threshold_deg: Max angle difference
        high_similarity_threshold: Above this similarity, trust appearance over motion

    Returns:
        List of validated (track_id, similarity) tuples (None for rejected)
    """
    validated = []

    for i, (track_id, sim) in enumerate(assignments):
        if track_id is None:
            validated.append((None, 0.0))
            continue

        # High-similarity matches: trust appearance over motion
        # RE-ID appearance match is more reliable than position prediction
        if sim >= high_similarity_threshold:
            validated.append((track_id, sim))
            continue

        # Skip validation for crossing tracks (expect erratic motion)
        if track_id in crossing_ids:
            validated.append((track_id, sim))
            continue

        # Skip if no prediction available
        if track_id not in predicted_positions:
            validated.append((track_id, sim))
            continue

        det_pos = detection_positions[i]
        pred_pos = predicted_positions[track_id]
        track_vel = track_velocities.get(track_id, (0.0, 0.0))

        # Implied velocity from prediction to detection
        implied_vel = (det_pos[0] - pred_pos[0], det_pos[1] - pred_pos[1])

        if validate_motion_consistency(
            det_pos,
            pred_pos,
            implied_vel,
            track_vel,
            max_distance=max_distance,
            direction_threshold_deg=direction_threshold_deg,
        ):
            validated.append((track_id, sim))
        else:
            validated.append((None, 0.0))  # Rejected

    return validated
