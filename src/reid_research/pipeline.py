"""Video ReID inference pipeline."""
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .config import ReIDConfig
from .feature_extractor import ReIDFeatureExtractor
from .gallery import PersonGallery
from .jointbdoe_detector import JointBDOEDetector, Detection
from .matching import compute_quality_score
from .visualization import (
    get_id_color,
    GalleryPanelRenderer,
    GalleryPanelEntry,
    HUDRenderer,
    ExtendedFrameRenderer,
    IDSwitchCapturer,
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

        # OSNet feature extractor (via torchreid)
        self.extractor = extractor or ReIDFeatureExtractor(config)
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

        # IoU-based track continuation: track_id -> (bbox, last_frame)
        self._track_bboxes: dict[int, tuple[tuple[float, ...], int]] = {}
        self._iou_threshold = 0.3  # Min IoU to continue track

        # Tentative track system: require N frames before permanent ID
        self._min_frames_for_id = config.gallery.min_frames_for_id
        self._tentative_max_age = config.gallery.tentative_max_age
        # tentative_id -> {bbox, features, frame_count, first_frame, crops}
        self._tentative_tracks: dict[int, dict] = {}
        self._next_tentative_id = -1  # Negative IDs for tentative tracks

        # Extended frame renderer (analytics outside video)
        self._extended_renderer = ExtendedFrameRenderer(config.visualization)
        self._recent_matches: list[dict] = []  # Track recent ID matches for bottom bar

        # ID switch capturer (initialized per video in process_video)
        self._id_switch_capturer: IDSwitchCapturer | None = None

    def process_frame(self, frame: np.ndarray) -> list[Detection]:
        """Process single frame: detect -> extract -> match.

        Uses batch matching to ensure unique IDs per frame.
        New detections must be seen for 5 frames before getting permanent ID.

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

        # Stage 2: IoU-based track continuation + ReID matching
        self._current_stage = 2
        features_list = [det.features for det in detections]
        bboxes = [det.bbox for det in detections]
        current_frame = self.gallery._frame_idx

        # First: try IoU-based continuation for permanent tracks (within 5 frames)
        iou_matched: dict[int, int] = {}  # det_idx -> track_id
        used_tracks = set()
        for det_idx, bbox in enumerate(bboxes):
            best_iou, best_tid = 0.0, None
            for tid, (prev_bbox, last_frame) in self._track_bboxes.items():
                if tid in used_tracks:
                    continue
                if current_frame - last_frame > 5:  # Skip stale tracks
                    continue
                iou = self._compute_iou(bbox, prev_bbox)
                if iou > best_iou and iou >= self._iou_threshold:
                    best_iou, best_tid = iou, tid
            if best_tid is not None:
                iou_matched[det_idx] = best_tid
                used_tracks.add(best_tid)

        # Try IoU continuation for tentative tracks
        tentative_iou_matched: dict[int, int] = {}  # det_idx -> tentative_id
        used_tentative = set()
        for det_idx, bbox in enumerate(bboxes):
            if det_idx in iou_matched:  # Already matched to permanent track
                continue
            best_iou, best_tid = 0.0, None
            for tid, track_data in self._tentative_tracks.items():
                if tid in used_tentative:
                    continue
                if current_frame - track_data["last_frame"] > 5:  # Skip stale
                    continue
                iou = self._compute_iou(bbox, track_data["bbox"])
                if iou > best_iou and iou >= self._iou_threshold:
                    best_iou, best_tid = iou, tid
            if best_tid is not None:
                tentative_iou_matched[det_idx] = best_tid
                used_tentative.add(best_tid)

        # Get recent IDs BEFORE match_batch updates temporal history
        recent_ids = [self.gallery.get_recent_id(bbox) for bbox in bboxes]

        # ReID matching for detections not matched by IoU
        results = self.gallery.match_batch(features_list, bboxes=bboxes)
        matched_ids = [r[0] for r in results]
        similarities = [r[1] for r in results]

        # Override with IoU matches where applicable (permanent tracks only)
        for det_idx, tid in iou_matched.items():
            if matched_ids[det_idx] is None:
                matched_ids[det_idx] = tid
                similarities[det_idx] = 0.9  # High confidence for IoU match

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
        for det_idx, (det, matched_id, recent_id, similarity, q_score) in enumerate(zip(
            detections, matched_ids, recent_ids, similarities, quality_scores
        )):
            det.match_similarity = similarity
            # Get top 3 similar IDs (excluding matched ID)
            det.top_similar = self.gallery.get_top_similar(
                det.features, exclude_id=matched_id, top_k=3
            )
            frame_idx = self.gallery._frame_idx

            if matched_id is not None:
                # ReID matched to existing gallery entry - assign permanent ID
                det.track_id = matched_id
                det.is_matched = True
                self._hud_renderer.add_event(f"ID#{det.track_id} matched", frame_idx, "match")
                self._recent_matches.append({"id": det.track_id, "similarity": similarity, "is_new": False})
                if len(self._recent_matches) > 20:
                    self._recent_matches.pop(0)

                if recent_id is None:
                    det.is_recovery = True

                if recent_id is not None and recent_id != matched_id:
                    det.previous_id = recent_id
                    prev_entry = self.gallery.get_entry(recent_id)
                    if prev_entry:
                        det.previous_id_timestamp = prev_entry.last_seen

                self.gallery.update(matched_id, det.features, quality_score=q_score, bbox=det.bbox)
                self._match_animations[matched_id] = 0
                self._track_bboxes[det.track_id] = (det.bbox, frame_idx)

                # If this detection was previously tentative, remove it
                if det_idx in tentative_iou_matched:
                    tent_id = tentative_iou_matched[det_idx]
                    if tent_id in self._tentative_tracks:
                        del self._tentative_tracks[tent_id]

            elif det_idx in tentative_iou_matched:
                # Continue existing tentative track
                tent_id = tentative_iou_matched[det_idx]
                track_data = self._tentative_tracks[tent_id]
                track_data["frame_count"] += 1
                track_data["bbox"] = det.bbox
                track_data["last_frame"] = frame_idx
                track_data["features"] = det.features  # Keep latest features
                track_data["crops"].append(det.crop)
                if len(track_data["crops"]) > 10:
                    track_data["crops"].pop(0)

                # Check if ready to promote to permanent ID
                if track_data["frame_count"] >= self._min_frames_for_id:
                    # Promote: add to gallery with accumulated features
                    det.track_id = self.gallery.add(
                        det.features, quality_score=q_score, bbox=det.bbox
                    )
                    det.is_matched = False
                    self._match_animations[det.track_id] = 0
                    self._hud_renderer.add_event(f"ID#{det.track_id} new", frame_idx, "new")
                    self._recent_matches.append({"id": det.track_id, "similarity": 0.0, "is_new": True})
                    if len(self._recent_matches) > 20:
                        self._recent_matches.pop(0)
                    self._track_bboxes[det.track_id] = (det.bbox, frame_idx)
                    # Remove from tentative
                    del self._tentative_tracks[tent_id]
                else:
                    # Still tentative - use negative ID for visualization
                    det.track_id = tent_id
                    det.is_matched = False

            else:
                # New detection - create tentative track
                tent_id = self._next_tentative_id
                self._next_tentative_id -= 1
                self._tentative_tracks[tent_id] = {
                    "bbox": det.bbox,
                    "features": det.features,
                    "frame_count": 1,
                    "first_frame": frame_idx,
                    "last_frame": frame_idx,
                    "crops": [det.crop],
                }
                det.track_id = tent_id  # Negative ID indicates tentative
                det.is_matched = False

        # Prune stale tentative tracks (not seen for >tentative_max_age frames)
        stale_tentative = [
            tid for tid, data in self._tentative_tracks.items()
            if current_frame - data["last_frame"] > self._tentative_max_age
        ]
        for tid in stale_tentative:
            del self._tentative_tracks[tid]

        return detections

    def _compute_iou(
        self,
        bbox1: tuple[float, ...],
        bbox2: tuple[float, ...],
    ) -> float:
        """Compute IoU between two bboxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        return inter / (area1 + area2 - inter)

    def _draw_dashed_rect(
        self,
        img: np.ndarray,
        pt1: tuple[int, int],
        pt2: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int = 2,
        dash_length: int = 10,
    ) -> None:
        """Draw a dashed rectangle on image."""
        x1, y1 = pt1
        x2, y2 = pt2
        # Top edge
        self._draw_dashed_line(img, (x1, y1), (x2, y1), color, thickness, dash_length)
        # Bottom edge
        self._draw_dashed_line(img, (x1, y2), (x2, y2), color, thickness, dash_length)
        # Left edge
        self._draw_dashed_line(img, (x1, y1), (x1, y2), color, thickness, dash_length)
        # Right edge
        self._draw_dashed_line(img, (x2, y1), (x2, y2), color, thickness, dash_length)

    def _draw_dashed_line(
        self,
        img: np.ndarray,
        pt1: tuple[int, int],
        pt2: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int = 2,
        dash_length: int = 10,
    ) -> None:
        """Draw a dashed line on image."""
        x1, y1 = pt1
        x2, y2 = pt2
        dx = x2 - x1
        dy = y2 - y1
        dist = max(1, int((dx**2 + dy**2) ** 0.5))
        for i in range(0, dist, dash_length * 2):
            start = i / dist
            end = min((i + dash_length) / dist, 1.0)
            sx = int(x1 + dx * start)
            sy = int(y1 + dy * start)
            ex = int(x1 + dx * end)
            ey = int(y1 + dy * end)
            cv2.line(img, (sx, sy), (ex, ey), color, thickness)

    def process_video(
        self,
        video_path: str | Path,
        output_path: str | Path | None = None,
        max_frames: int | None = None,
    ) -> dict:
        """Process video file.

        Args:
            video_path: Input video path
            output_path: Output video path (optional)
            max_frames: Maximum frames to process (optional)

        Returns:
            Processing statistics dict
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Initialize ID switch capturer if enabled
        if self.config.debug.capture_id_switches:
            output_dir = Path(self.config.debug.id_switch_output_dir) / video_path.stem
            self._id_switch_capturer = IDSwitchCapturer(
                output_dir=output_dir,
                frames_before=self.config.debug.id_switch_frames_before,
                frames_after=self.config.debug.id_switch_frames_after,
                enabled=True,
            )
        else:
            self._id_switch_capturer = None

        # Video properties
        self.fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames is not None:
            total_frames = min(total_frames, max_frames)

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
            if not ret or (max_frames is not None and stats["frames"] >= max_frames):
                break

            detections = self.process_frame(frame)
            stats["frames"] += 1
            stats["detections"] += len(detections)

            # Push frame to ID switch capturer (if enabled)
            if self._id_switch_capturer:
                self._id_switch_capturer.push_frame(frame, stats["frames"], detections)

            # Count ReID matches and unique IDs (exclude tentative tracks with negative IDs)
            frame_matches = 0
            for det in detections:
                if det.track_id is not None and det.track_id >= 0:
                    stats["unique_ids"].add(det.track_id)
                if det.is_matched:
                    frame_matches += 1
                    stats["reid_matches"] += 1

            # Visualize and write
            if writer and self.config.output.visualization:
                if self.config.visualization.extended_frame_enabled:
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

        # Flush any pending ID switch captures
        if self._id_switch_capturer:
            self._id_switch_capturer.flush()
            if self._id_switch_capturer.switch_count > 0:
                print(f"Captured {self._id_switch_capturer.switch_count} ID switches to {self._id_switch_capturer._output_dir}")

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

        # Update thumbnail cache and collect active IDs (skip tentative tracks)
        for det in detections:
            if det.track_id is not None and det.track_id >= 0:
                active_ids.add(det.track_id)
                self._update_thumbnail_cache(det)

        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            track_id = det.track_id if det.track_id is not None else -1

            # Skip tentative tracks (negative IDs) in visualization
            # Or show them with a distinct style (dashed/gray)
            is_tentative = track_id < 0

            # Consistent color per ID using Okabe-Ito palette
            # Use gray for tentative tracks
            color = (128, 128, 128) if is_tentative else get_id_color(track_id)

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

            # Draw box border (dashed for tentative)
            if is_tentative:
                # Draw dashed rectangle for tentative tracks
                self._draw_dashed_rect(vis, (x1, y1), (x2, y2), color, thickness)
            else:
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

            # Modified label logic: Move ID to middle, remove status labels
            # Show "?" for tentative tracks
            label = "?" if is_tentative else f"ID:{track_id:02d}"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Draw label background and text in the middle
            cv2.rectangle(vis, (cx - tw // 2 - 4, cy - th // 2 - 4), (cx + tw // 2 + 4, cy + th // 2 + 4), color, -1)
            cv2.putText(
                vis, label, (cx - tw // 2, cy + th // 2 + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

            # Show top 3 similar IDs below bbox (skip for tentative tracks)
            if det.top_similar and not is_tentative:
                y_offset = y2 + 15
                for sim_id, sim_score in det.top_similar[:3]:
                    sim_label = f"~ID:{sim_id:02d}: {sim_score*100:.0f}%"
                    (sw, sh), _ = cv2.getTextSize(sim_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    # Draw below bbox
                    cv2.rectangle(vis, (x1, y_offset - sh - 2), (x1 + sw + 6, y_offset + 2), (40, 40, 40), -1)
                    cv2.putText(vis, sim_label, (x1 + 3, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 255), 1)
                    y_offset += sh + 6

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

