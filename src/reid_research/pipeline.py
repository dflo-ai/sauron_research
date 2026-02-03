"""Video ReID inference pipeline."""
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .config import ReIDConfig
from .detector import Detection
from .feature_extractor import ReIDFeatureExtractor
from .gallery import PersonGallery
from .jointbdoe_detector import JointBDOEDetector
from .matching import compute_quality_score
from .visualization import (
    get_id_color,
    GalleryPanelRenderer,
    GalleryPanelEntry,
    HUDRenderer,
    SplitViewRenderer,
    ExtendedFrameRenderer,
)


class VideoReIDPipeline:
    """End-to-end video person re-identification pipeline."""

    # Animation settings
    FADE_DURATION = 15  # Frames for opacity fade from 50% to 0%
    REMATCH_DISPLAY_SECONDS = 3  # How long to show rematch info

    def __init__(
        self,
        config: ReIDConfig,
        detector: JointBDOEDetector | None = None,
        extractor: ReIDFeatureExtractor | None = None,
        gallery: PersonGallery | None = None,
    ):
        """Initialize pipeline with config and optional components.

        Args:
            config: ReIDConfig instance
            detector: Optional JointBDOEDetector (created if None)
            extractor: Optional ReIDFeatureExtractor (created if None)
            gallery: Optional PersonGallery (created if None)
        """
        self.config = config

        # Use JointBDOE detector (person detection + ReID features in one model)
        self.detector = detector or JointBDOEDetector(config)

        # Select extractor based on config: FastReID or torchreid OSNet
        if extractor is not None:
            self.extractor = extractor
        elif config.model.use_fastreid:
            # Lazy import to avoid fastreid dependency when not used
            from .fastreid_extractor import FastReIDExtractor
            self.extractor = FastReIDExtractor(config)
        else:
            self.extractor = ReIDFeatureExtractor(config)
        self.gallery = gallery or PersonGallery(config)
        self.fps = 30  # Default FPS

        # Animation state: track_id -> frames_since_match
        self._match_animations: dict[int, int] = {}

        # Thumbnail cache for gallery panel: track_id -> (thumbnail, first_seen, det_count)
        self._thumbnail_cache: dict[int, tuple[np.ndarray, int, int]] = {}
        self._thumbnail_size = (60, 80)  # W, H

        # Gallery panel renderer
        self._gallery_renderer = GalleryPanelRenderer(config.visualization)

        # HUD renderer for pipeline stages and stats
        self._hud_renderer = HUDRenderer(config.visualization)
        self._current_stage = 0  # Pipeline stage: 0=Detect, 1=Extract, 2=Match, 3=Track

        # Split view renderer
        self._split_view_renderer = SplitViewRenderer(config.visualization)

        # Extended frame renderer (analytics outside video)
        self._extended_renderer = ExtendedFrameRenderer(config.visualization)
        self._recent_matches: list[dict] = []  # Track recent ID matches for bottom bar

    def process_frame(self, frame: np.ndarray) -> list[Detection]:
        """Process single frame: detect -> extract -> match.

        Uses batch matching to ensure unique IDs per frame.

        Args:
            frame: BGR numpy array

        Returns:
            List of detections with assigned track_ids
        """
        # Stage 0: Detect persons
        self._current_stage = 0
        detections = self.detector.detect(frame)

        if not detections:
            return []

        # Stage 1: Extract ReID features
        self._current_stage = 1
        crops = [d.crop for d in detections]
        features = self.extractor.extract(crops)

        # Store features in detections
        for det, feat in zip(detections, features):
            det.features = feat

        # Stage 2: Batch match to ensure unique IDs per frame
        self._current_stage = 2
        features_list = [det.features for det in detections]
        bboxes = [det.bbox for det in detections]

        # Get recent IDs BEFORE match_batch updates temporal history
        recent_ids = [self.gallery.get_recent_id(bbox) for bbox in bboxes]

        results = self.gallery.match_batch(features_list, bboxes=bboxes)
        matched_ids = [r[0] for r in results]
        similarities = [r[1] for r in results]

        # Compute quality scores for each detection
        gallery_cfg = self.config.gallery
        quality_scores = []
        for det in detections:
            if gallery_cfg.use_quality_weighting:
                q_score = compute_quality_score(
                    confidence=det.confidence,
                    bbox=det.bbox,
                    min_bbox_area=gallery_cfg.min_bbox_area,
                    ideal_aspect_ratio=gallery_cfg.ideal_aspect_ratio,
                    confidence_weight=gallery_cfg.quality_confidence_weight,
                    geometry_weight=gallery_cfg.quality_geometry_weight,
                )
            else:
                q_score = 1.0
            quality_scores.append(q_score)

        # Stage 3: Assign IDs and update gallery (Track)
        self._current_stage = 3
        for det, matched_id, recent_id, similarity, q_score in zip(
            detections, matched_ids, recent_ids, similarities, quality_scores
        ):
            det.match_similarity = similarity
            frame_idx = self.gallery._frame_idx
            if matched_id is None:
                det.track_id = self.gallery.add(
                    det.features, quality_score=q_score, bbox=det.bbox
                )
                det.is_matched = False  # New person
                # Start animation for new ID
                self._match_animations[det.track_id] = 0
                # Add event to ticker
                self._hud_renderer.add_event(f"ID#{det.track_id} new", frame_idx, "new")
                # Track for extended frame bottom bar
                self._recent_matches.append({"id": det.track_id, "similarity": 0.0, "is_new": True})
                if len(self._recent_matches) > 20:
                    self._recent_matches.pop(0)
            else:
                det.track_id = matched_id
                det.is_matched = True  # ReID match found
                # Add event to ticker
                self._hud_renderer.add_event(f"ID#{det.track_id} matched", frame_idx, "match")
                # Track for extended frame bottom bar (estimate similarity)
                self._recent_matches.append({"id": det.track_id, "similarity": similarity, "is_new": False})
                if len(self._recent_matches) > 20:
                    self._recent_matches.pop(0)

                # Set recovery flag if track was lost spatially
                if recent_id is None:
                    det.is_recovery = True

                # Detect if this is a rematch from a DIFFERENT previous ID
                if recent_id is not None and recent_id != matched_id:
                    det.previous_id = recent_id
                    prev_entry = self.gallery.get_entry(recent_id)
                    if prev_entry:
                        det.previous_id_timestamp = prev_entry.last_seen

                self.gallery.update(
                    matched_id, det.features, quality_score=q_score, bbox=det.bbox
                )
                # Reset animation on re-match
                self._match_animations[matched_id] = 0

        return detections

    def process_video(
        self,
        video_path: str | Path,
        output_path: str | Path | None = None,
    ) -> dict:
        """Process entire video file.

        Args:
            video_path: Input video path
            output_path: Output video path (optional)

        Returns:
            Processing statistics dict
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Video properties
        self.fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output writer - use extended dimensions if enabled
        writer = None
        if output_path and self.config.output.save_video:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            if self.config.visualization.extended_frame_enabled:
                ext_w, ext_h = self._extended_renderer.get_extended_dimensions(width, height)
                writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (ext_w, ext_h))
            else:
                writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, height))

        # Processing loop with progress bar
        stats = {"frames": 0, "detections": 0, "unique_ids": set(), "reid_matches": 0}

        pbar = tqdm(total=total_frames, desc="Processing", unit="frame")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.process_frame(frame)
            stats["frames"] += 1
            stats["detections"] += len(detections)

            # Count ReID matches and unique IDs
            frame_matches = 0
            for det in detections:
                if det.track_id is not None:
                    stats["unique_ids"].add(det.track_id)
                if det.is_matched:
                    frame_matches += 1
                    stats["reid_matches"] += 1

            # Visualize and write
            if writer and self.config.output.visualization:
                # Check if split view is enabled
                if self.config.visualization.split_view_enabled:
                    stage_frames = self._generate_stage_frames(frame, detections)
                    vis_frame = self._split_view_renderer.render(
                        stage_frames, (width, height)
                    )
                elif self.config.visualization.extended_frame_enabled:
                    # Extended layout: analytics outside video frame
                    vis_frame = self._visualize(frame, detections)
                    gallery_entries = self._get_gallery_entries(detections)
                    vis_frame = self._extended_renderer.create_extended_frame(
                        video_frame=vis_frame,
                        gallery_entries=gallery_entries,
                        stats={
                            "frame_idx": stats["frames"],
                            "total_frames": total_frames,
                            "detections": len(detections),
                            "reid_matches": frame_matches,
                            "unique_ids": len(stats["unique_ids"]),
                        },
                        recent_matches=self._recent_matches[-10:],
                        fps=self.fps,
                    )
                else:
                    vis_frame = self._visualize(frame, detections)
                    # Add HUD overlay on video (legacy mode)
                    if self.config.visualization.show_pipeline_hud:
                        vis_frame = self._draw_hud(
                            vis_frame, stats["frames"], total_frames,
                            len(detections), frame_matches, len(stats["unique_ids"])
                        )
                writer.write(vis_frame)

            # Step gallery frame counter
            self.gallery.step_frame()

            # Prune stale entries periodically
            if stats["frames"] % 300 == 0:
                self.gallery.prune_stale(max_age=300)

            # Update progress bar with ReID stats
            pbar.set_postfix({
                "det": len(detections),
                "reid": frame_matches,
                "ids": len(stats["unique_ids"])
            })
            pbar.update(1)

        pbar.close()

        cap.release()
        if writer:
            writer.release()

        stats["unique_ids"] = len(stats["unique_ids"])
        return stats

    def _format_time(self, frame_idx: int) -> str:
        """Format frame index as MM:SS."""
        total_seconds = int(frame_idx / self.fps)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    def _visualize(
        self, frame: np.ndarray, detections: list[Detection]
    ) -> np.ndarray:
        """Draw bounding boxes and track IDs with animated ReID effects.

        Shows fading fill effect (50%->0%) when ReID match/init occurs.
        Flashes the bbox when a rematch is detected.

        Args:
            frame: BGR numpy array
            detections: List of detections with track_ids

        Returns:
            Annotated frame
        """
        vis = frame.copy()
        active_ids = set()

        # Update thumbnail cache and collect active IDs
        for det in detections:
            if det.track_id is not None:
                active_ids.add(det.track_id)
                self._update_thumbnail_cache(det)

        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            track_id = det.track_id if det.track_id is not None else -1

            # Consistent color per ID using Okabe-Ito palette
            color = get_id_color(track_id)

            # Animated fill effect (fades from 30% to 0% over FADE_DURATION frames)
            anim_frame = self._match_animations.get(track_id, self.FADE_DURATION)
            
            # Flash effect for rematch
            is_rematch = det.previous_id is not None
            if is_rematch and anim_frame < self.FADE_DURATION:
                # Flash by oscillating color or thickness
                if (anim_frame // 3) % 2 == 0:
                    color = (255, 255, 255)  # Flash white
                    thickness = 6
                else:
                    thickness = 4
            else:
                thickness = 2 if anim_frame >= self.FADE_DURATION else (4 if anim_frame == 0 else 3)

            if anim_frame < self.FADE_DURATION:
                # Calculate fading opacity: 0.3 -> 0.0
                opacity = 0.3 * (1 - anim_frame / self.FADE_DURATION)
                overlay = vis.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                vis = cv2.addWeighted(overlay, opacity, vis, 1 - opacity, 0)

            # Draw box border
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

            # Modified label logic: Move ID to middle, remove status labels
            label = f"ID:{track_id:02d}"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Draw label background and text in the middle
            cv2.rectangle(vis, (cx - tw // 2 - 4, cy - th // 2 - 4), (cx + tw // 2 + 4, cy + th // 2 + 4), color, -1)
            cv2.putText(
                vis, label, (cx - tw // 2, cy + th // 2 + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

            # Show similarity if recovery: "similar to ID:XX n%"
            rematch_frames = self.fps * self.REMATCH_DISPLAY_SECONDS
            if det.is_recovery and anim_frame < rematch_frames:
                recovery_label = f"similar to ID:{track_id:02d} {det.match_similarity*100:.0f}%"
                (rw, rh), _ = cv2.getTextSize(recovery_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                # Draw below the main ID in the middle
                cv2.rectangle(vis, (cx - rw // 2 - 4, cy + th // 2 + 5), (cx + rw // 2 + 4, cy + th // 2 + rh + 12), (0, 0, 0), -1)
                cv2.putText(
                    vis, recovery_label, (cx - rw // 2, cy + th // 2 + rh + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1
                )

            # Draw rematch info: [ID:10 (00:10) --> ID:01 (00:11)]
            # Show rematch info for REMATCH_DISPLAY_SECONDS
            rematch_frames = self.fps * self.REMATCH_DISPLAY_SECONDS
            if is_rematch and anim_frame < rematch_frames:
                prev_time = self._format_time(det.previous_id_timestamp or 0)
                curr_time = self._format_time(self.gallery._frame_idx)
                rematch_label = f"[ID:{det.previous_id:02d} ({prev_time}) --> ID:{track_id:02d} ({curr_time})]"
                
                (rtw, rth), _ = cv2.getTextSize(rematch_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                # Draw directly above bbox since main label moved to middle
                cv2.rectangle(vis, (x1, y1 - rth - 10), (x1 + rtw + 4, y1), (0, 0, 0), -1)
                cv2.putText(
                    vis, rematch_label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
                )

        # Step animation counters
        for tid in list(self._match_animations.keys()):
            self._match_animations[tid] += 1
            # Cleanup old animations
            # Keep animation state for rematch display duration
            if self._match_animations[tid] > self.fps * self.REMATCH_DISPLAY_SECONDS:
                del self._match_animations[tid]

        # Draw gallery panel if enabled (only for non-extended mode)
        if self.config.visualization.show_gallery_panel and not self.config.visualization.extended_frame_enabled:
            vis = self._draw_gallery_panel(vis, active_ids)

        return vis

    def _get_gallery_entries(self, detections: list[Detection]) -> list[GalleryPanelEntry]:
        """Get gallery entries for extended frame renderer.

        Args:
            detections: Current frame detections

        Returns:
            List of GalleryPanelEntry sorted by last_seen
        """
        active_ids = {det.track_id for det in detections if det.track_id is not None}
        entries = []

        for track_id in self.gallery.get_all_ids():
            if track_id not in self._thumbnail_cache:
                continue

            thumb, first_seen, det_count = self._thumbnail_cache[track_id]
            gallery_entry = self.gallery.get_entry(track_id)
            if gallery_entry is None:
                continue

            entries.append(
                GalleryPanelEntry(
                    track_id=track_id,
                    thumbnail=thumb,
                    first_seen=first_seen,
                    last_seen=gallery_entry.last_seen,
                    detection_count=det_count,
                    is_active=track_id in active_ids,
                )
            )

        # Sort by last_seen descending
        entries.sort(key=lambda e: e.last_seen, reverse=True)
        return entries

    def _update_thumbnail_cache(self, det: Detection) -> None:
        """Update thumbnail cache for a detection.

        Args:
            det: Detection with crop and track_id
        """
        if det.track_id is None or det.crop is None:
            return

        track_id = det.track_id
        frame_idx = self.gallery._frame_idx

        if track_id in self._thumbnail_cache:
            # Update existing: keep first_seen, increment count
            _, first_seen, count = self._thumbnail_cache[track_id]
            thumb = cv2.resize(det.crop, self._thumbnail_size)
            self._thumbnail_cache[track_id] = (thumb, first_seen, count + 1)
        else:
            # New entry
            thumb = cv2.resize(det.crop, self._thumbnail_size)
            self._thumbnail_cache[track_id] = (thumb, frame_idx, 1)

    def _draw_gallery_panel(
        self, frame: np.ndarray, active_ids: set[int]
    ) -> np.ndarray:
        """Draw gallery side panel with person thumbnails.

        Args:
            frame: Main video frame
            active_ids: Set of track IDs visible in current frame

        Returns:
            Frame with gallery panel overlaid
        """
        # Build gallery entries sorted by last_seen (most recent first)
        entries = []
        for track_id in self.gallery.get_all_ids():
            if track_id not in self._thumbnail_cache:
                continue

            thumb, first_seen, det_count = self._thumbnail_cache[track_id]
            gallery_entry = self.gallery.get_entry(track_id)
            if gallery_entry is None:
                continue

            entries.append(
                GalleryPanelEntry(
                    track_id=track_id,
                    thumbnail=thumb,
                    first_seen=first_seen,
                    last_seen=gallery_entry.last_seen,
                    detection_count=det_count,
                    is_active=track_id in active_ids,
                )
            )

        # Sort by last_seen descending (most recent first)
        entries.sort(key=lambda e: e.last_seen, reverse=True)

        # Render panel
        panel = self._gallery_renderer.render(frame.shape[0], entries, self.fps)

        # Composite onto frame
        return self._gallery_renderer.composite_on_frame(frame, panel)

    def _draw_hud(
        self, frame: np.ndarray, frame_idx: int, total_frames: int,
        detections: int, reid_matches: int, unique_ids: int
    ) -> np.ndarray:
        """Draw HUD with pipeline stages, stats, and event ticker.

        Args:
            frame: BGR numpy array
            frame_idx: Current frame number
            total_frames: Total frames in video
            detections: Number of detections this frame
            reid_matches: Number of ReID matches this frame
            unique_ids: Total unique IDs so far

        Returns:
            Frame with HUD overlay
        """
        h, w = frame.shape[:2]

        # Render HUD components
        pipeline_bar = self._hud_renderer.render_pipeline_bar(w, self._current_stage)
        stats_panel = self._hud_renderer.render_stats_panel(
            fps=self.fps,
            frame_idx=frame_idx,
            total_frames=total_frames,
            detections=detections,
            reid_matches=reid_matches,
            unique_ids=unique_ids,
            threshold=self.gallery.get_effective_threshold(),
        )
        event_ticker = self._hud_renderer.render_event_ticker(w, self.gallery._frame_idx)

        # Composite all HUD elements
        return self._hud_renderer.composite_hud(frame, pipeline_bar, stats_panel, event_ticker)

    def _draw_stats_overlay(
        self, frame: np.ndarray, frame_idx: int, total_frames: int,
        detections: int, reid_matches: int, unique_ids: int
    ) -> np.ndarray:
        """Draw stats overlay on frame (top-left corner).

        Args:
            frame: BGR numpy array
            frame_idx: Current frame number
            total_frames: Total frames in video
            detections: Number of detections this frame
            reid_matches: Number of ReID matches this frame
            unique_ids: Total unique IDs so far

        Returns:
            Frame with stats overlay
        """
        vis = frame.copy()

        # Stats text
        lines = [
            f"Frame: {frame_idx}/{total_frames}",
            f"Detections: {detections}",
            f"ReID Matches: {reid_matches}",
            f"Unique IDs: {unique_ids}",
        ]

        # Draw semi-transparent background
        padding = 10
        line_height = 25
        box_h = len(lines) * line_height + padding * 2
        box_w = 200

        overlay = vis.copy()
        cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)
        vis = cv2.addWeighted(overlay, 0.6, vis, 0.4, 0)

        # Draw text
        y = 10 + padding + 18
        for line in lines:
            # Highlight ReID matches in green
            color = (0, 255, 0) if "ReID" in line and reid_matches > 0 else (255, 255, 255)
            cv2.putText(vis, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += line_height

        return vis

    def _generate_stage_frames(
        self, frame: np.ndarray, detections: list[Detection]
    ) -> dict[str, np.ndarray]:
        """Generate visualization frames for each pipeline stage.

        Args:
            frame: Original BGR frame
            detections: List of detections with track_ids

        Returns:
            Dict mapping stage name to frame
        """
        stages = {}
        stages["original"] = frame.copy()
        stages["detection"] = self._draw_detections_only(frame, detections)
        stages["reid"] = self._visualize(frame, detections)
        return stages

    def _draw_detections_only(
        self, frame: np.ndarray, detections: list[Detection]
    ) -> np.ndarray:
        """Draw only bounding boxes without IDs (detection stage view).

        Args:
            frame: BGR numpy array
            detections: List of detections

        Returns:
            Annotated frame with boxes only
        """
        vis = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)

            # Simple white boxes for detection stage
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # Confidence label
            conf_label = f"{det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), (255, 255, 255), -1)
            cv2.putText(
                vis, conf_label, (x1 + 2, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

        return vis
