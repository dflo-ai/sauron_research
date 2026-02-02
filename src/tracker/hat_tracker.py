"""HAT-enhanced multi-object tracker with ReID embeddings.

Implements History-Aware Transformation for improved identity association.
Uses appearance-only matching (pure ReID) as per project requirements.
"""

from dataclasses import dataclass, field
from typing import Literal
import torch
from torch import Tensor
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ..hat_reid.lda import LDA
from ..hat_reid.queues import FIFOQueue, ScoreQueue


@dataclass
class Track:
    """Single track state."""
    id: int
    bbox: Tensor                           # (4,) xyxy format
    embed: Tensor                          # (D,) averaged embedding
    score: float                           # Detection confidence
    last_frame: int                        # Last seen frame
    history: FIFOQueue | ScoreQueue = field(default_factory=lambda: FIFOQueue())


class HATTracker:
    """History-Aware Transformation tracker with ReID embeddings.

    Uses appearance-only matching (no IoU) based on embedding similarity.
    HAT transformation activates when sufficient history is accumulated.
    """

    def __init__(
        self,
        # Track management
        init_score_thr: float = 0.8,
        match_score_thr: float = 0.5,
        max_lost_frames: int = 10,
        memo_momentum: float = 0.8,
        # HAT config
        use_hat: bool = True,
        hat_factor_thr: float = 4.0,
        history_max_len: int = 60,
        history_decay: float = 0.9,
        queue_type: Literal["fifo", "score"] = "fifo",
        # LDA config
        use_shrinkage: bool = True,
        use_weighted_class_mean: bool = True,
        # Matching
        similarity_mode: Literal["cosine", "bisoftmax", "masa"] = "cosine",
        # Device
        device: str = "cuda",
    ):
        """Initialize HAT tracker.

        Args:
            init_score_thr: Minimum score to initialize new track
            match_score_thr: Minimum similarity score for association
            max_lost_frames: Frames before removing lost track
            memo_momentum: EMA factor for embedding update (higher = more new)
            use_hat: Enable HAT transformation
            hat_factor_thr: History/tracks ratio to activate HAT
            history_max_len: Max samples per track history
            history_decay: Weight decay ratio for history queue
            queue_type: "fifo" or "score" queue type
            use_shrinkage: Use shrinkage in LDA covariance
            use_weighted_class_mean: Weight LDA class means by scores
            similarity_mode: Similarity computation method
            device: Computation device (must be cuda per requirements)
        """
        self.init_score_thr = init_score_thr
        self.match_score_thr = match_score_thr
        self.max_lost_frames = max_lost_frames
        self.memo_momentum = memo_momentum

        self.use_hat = use_hat
        self.hat_factor_thr = hat_factor_thr
        self.history_max_len = history_max_len
        self.history_decay = history_decay
        self.queue_type = queue_type

        self.use_shrinkage = use_shrinkage
        self.use_weighted_class_mean = use_weighted_class_mean

        self.similarity_mode = similarity_mode
        self.device = torch.device(device)

        self.tracks: dict[int, Track] = {}
        self.next_id = 0

        # Track HAT activation for metrics
        self._last_hat_active = False

    def update(
        self,
        boxes: Tensor,       # (N, 4) xyxy
        scores: Tensor,      # (N,)
        embeddings: Tensor,  # (N, D)
        frame_id: int,
    ) -> Tensor:
        """Process frame detections and return track IDs.

        Args:
            boxes: Detection bounding boxes in xyxy format
            scores: Detection confidence scores
            embeddings: ReID feature embeddings
            frame_id: Current frame number

        Returns:
            Track IDs for each detection, -1 for unmatched/new
        """
        boxes = boxes.to(self.device)
        scores = scores.to(self.device)
        embeddings = embeddings.to(self.device)

        if len(boxes) == 0:
            self._cleanup_lost_tracks(frame_id)
            return torch.empty(0, dtype=torch.long, device=self.device)

        ids = torch.full((len(boxes),), -1, dtype=torch.long, device=self.device)

        if self.tracks:
            # Get memo from active tracks
            memo = self._get_memo()

            if len(memo["embeds"]) > 0:
                # Compute similarity matrix (appearance-only)
                match_scores = self._compute_similarity(
                    embeddings, memo["embeds"], memo["hist_embeds"],
                    memo["hist_scores"], memo["hist_ids"]
                )

                # Hungarian assignment
                ids = self._assign_hungarian(
                    ids, match_scores, scores, memo["ids"]
                )

        # Update matched tracks, create new ones
        self._update_tracks(ids, boxes, scores, embeddings, frame_id)
        self._cleanup_lost_tracks(frame_id)

        return ids

    def _compute_similarity(
        self,
        det_embeds: Tensor,
        track_embeds: Tensor,
        hist_embeds: Tensor,
        hist_scores: Tensor,
        hist_ids: Tensor,
    ) -> Tensor:
        """Compute detection-to-track similarity matrix.

        Uses appearance-only matching (pure ReID) per project requirements.
        """
        # Check if HAT should be applied
        n_tracks = len(track_embeds)
        n_history = len(hist_embeds) if len(hist_embeds) > 0 else 0

        self._last_hat_active = (
            self.use_hat
            and n_history > self.hat_factor_thr * n_tracks
            and n_tracks >= 2  # Need at least 2 classes for LDA
        )

        if self._last_hat_active:
            lda = LDA(
                use_shrinkage=self.use_shrinkage,
                use_weighted_class_mean=self.use_weighted_class_mean,
                device=self.device,
            )
            lda.fit(hist_embeds, hist_ids, scores=hist_scores)

            if lda.is_fitted():
                det_t = lda.transform(det_embeds)
                track_t = lda.transform(track_embeds)
            else:
                det_t, track_t = det_embeds, track_embeds
                self._last_hat_active = False
        else:
            det_t, track_t = det_embeds, track_embeds

        # Compute similarity based on mode
        if self.similarity_mode == "cosine":
            return self._cosine_similarity(det_t, track_t)
        elif self.similarity_mode == "bisoftmax":
            return self._bisoftmax_similarity(det_t, track_t)
        else:  # masa
            sim_cos = self._cosine_similarity(det_t, track_t)
            sim_bi = self._bisoftmax_similarity(det_t, track_t)
            return (sim_cos + sim_bi) / 2

    def _cosine_similarity(self, det_embeds: Tensor, track_embeds: Tensor) -> Tensor:
        """Compute cosine similarity matrix."""
        det_norm = F.normalize(det_embeds, p=2, dim=1)
        track_norm = F.normalize(track_embeds, p=2, dim=1)
        return det_norm @ track_norm.T

    def _bisoftmax_similarity(self, det_embeds: Tensor, track_embeds: Tensor) -> Tensor:
        """Compute bi-softmax similarity (symmetric softmax)."""
        feats = det_embeds @ track_embeds.T
        d2t = F.softmax(feats, dim=1)
        t2d = F.softmax(feats, dim=0)
        return (d2t + t2d) / 2

    def _assign_hungarian(
        self,
        ids: Tensor,
        match_scores: Tensor,
        det_scores: Tensor,
        track_ids: list[int],
    ) -> Tensor:
        """Optimal assignment via Hungarian algorithm."""
        cost = -match_scores.cpu().numpy()
        row_idx, col_idx = linear_sum_assignment(cost)

        for r, c in zip(row_idx, col_idx):
            if match_scores[r, c] > self.match_score_thr:
                ids[r] = track_ids[c]

        return ids

    def _get_memo(self) -> dict:
        """Collect memory from active tracks."""
        embeds, bboxes, ids = [], [], []
        hist_embeds, hist_scores, hist_ids = [], [], []

        for tid, track in self.tracks.items():
            embeds.append(track.embed)
            bboxes.append(track.bbox)
            ids.append(tid)

            feats, weights = track.history.get_features_and_weights()
            if len(feats) > 0:
                hist_embeds.append(feats)
                hist_scores.append(weights)
                hist_ids.extend([tid] * len(feats))

        result = {
            "embeds": torch.stack(embeds) if embeds else torch.empty(0, device=self.device),
            "bboxes": torch.stack(bboxes) if bboxes else torch.empty(0, device=self.device),
            "ids": ids,
            "hist_embeds": torch.cat(hist_embeds) if hist_embeds else torch.empty(0, device=self.device),
            "hist_scores": torch.cat(hist_scores) if hist_scores else torch.empty(0, device=self.device),
            "hist_ids": torch.tensor(hist_ids, dtype=torch.long, device=self.device),
        }
        return result

    def _update_tracks(
        self, ids: Tensor, boxes: Tensor, scores: Tensor,
        embeds: Tensor, frame_id: int
    ) -> None:
        """Update existing tracks and create new ones."""
        for i, (tid, box, score, emb) in enumerate(zip(ids, boxes, scores, embeds)):
            tid_val = int(tid.item())
            score_val = float(score.item())

            if tid_val >= 0 and tid_val in self.tracks:
                # Update existing track
                t = self.tracks[tid_val]
                t.bbox = box
                # EMA update: new_embed = (1-momentum)*old + momentum*new
                t.embed = (1 - self.memo_momentum) * t.embed + self.memo_momentum * emb
                t.score = score_val
                t.last_frame = frame_id
                t.history.add(emb, score_val)

            elif tid_val == -1 and score_val > self.init_score_thr:
                # Create new track
                if self.queue_type == "fifo":
                    queue = FIFOQueue(self.history_max_len, self.history_decay)
                else:
                    queue = ScoreQueue(self.history_max_len, self.history_decay)
                queue.add(emb, score_val)

                self.tracks[self.next_id] = Track(
                    id=self.next_id,
                    bbox=box,
                    embed=emb.clone(),
                    score=score_val,
                    last_frame=frame_id,
                    history=queue,
                )
                ids[i] = self.next_id
                self.next_id += 1

    def _cleanup_lost_tracks(self, frame_id: int) -> None:
        """Remove tracks not seen for too long."""
        lost = [
            tid for tid, t in self.tracks.items()
            if frame_id - t.last_frame >= self.max_lost_frames
        ]
        for tid in lost:
            del self.tracks[tid]

    def reset(self) -> None:
        """Clear all tracks and reset state."""
        self.tracks.clear()
        self.next_id = 0
        self._last_hat_active = False

    @property
    def num_tracks(self) -> int:
        """Number of active tracks."""
        return len(self.tracks)

    @property
    def hat_active(self) -> bool:
        """Whether HAT was active on last update."""
        return self._last_hat_active
