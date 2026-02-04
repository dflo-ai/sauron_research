"""ID Switch Frame Capturer - captures frames when Re-ID switches occur for debugging."""
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .colors import get_id_color

if TYPE_CHECKING:
    from ..jointbdoe_detector import Detection


@dataclass
class FrameData:
    """Single frame with metadata for buffer storage."""

    frame: np.ndarray  # BGR image (copy)
    frame_idx: int
    detections: list  # Detection objects for annotations


@dataclass
class PendingCapture:
    """Queued capture waiting for after-frames."""

    old_id: int
    new_id: int
    switch_frame_idx: int
    frames_before: list[FrameData]
    frames_after: list[FrameData] = field(default_factory=list)
    switch_frame: FrameData | None = None
    switch_bbox: tuple[float, ...] | None = None  # bbox of detection that switched


class IDSwitchCapturer:
    """Captures frames around ID switch events for manual investigation."""

    def __init__(
        self,
        output_dir: Path,
        frames_before: int = 3,
        frames_after: int = 3,
        enabled: bool = True,
    ):
        """Initialize capturer.

        Args:
            output_dir: Directory to save captured images
            frames_before: Number of frames to capture before switch
            frames_after: Number of frames to capture after switch
            enabled: Whether to capture (no-op if False)
        """
        self._output_dir = Path(output_dir)
        self._frames_before = frames_before
        self._frames_after = frames_after
        self._enabled = enabled

        # Rolling buffer of recent frames
        self._frame_buffer: deque[FrameData] = deque(maxlen=frames_before)

        # Pending captures waiting for after-frames
        self._pending_captures: list[PendingCapture] = []

        # Track switch count for stats
        self._switch_count = 0

    def push_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        detections: list["Detection"],
    ) -> None:
        """Push frame to buffer and check for ID switches.

        Args:
            frame: BGR image
            frame_idx: Current frame number
            detections: List of Detection objects with track_ids
        """
        if not self._enabled:
            return

        # Create frame data with copy (frames may be reused)
        frame_data = FrameData(
            frame=frame.copy(),
            frame_idx=frame_idx,
            detections=list(detections),
        )

        # Check for ID switches in detections
        switches = self._check_for_switches(detections)
        for old_id, new_id, det in switches:
            self._create_pending_capture(old_id, new_id, frame_data, det)

        # Update pending captures with after-frames
        self._update_pending_captures(frame_data)

        # Add to buffer (after processing to not include switch frame in before-frames)
        self._frame_buffer.append(frame_data)

    def _check_for_switches(
        self,
        detections: list["Detection"],
    ) -> list[tuple[int, int, "Detection"]]:
        """Check detections for ID switches.

        Args:
            detections: List of Detection objects

        Returns:
            List of (old_id, new_id, detection) tuples for switches
        """
        switches = []
        for det in detections:
            # Only capture switches between permanent IDs (>= 0)
            # Skip tentative tracks (negative IDs) to avoid false positives
            if (det.previous_id is not None and det.track_id is not None
                    and det.previous_id >= 0 and det.track_id >= 0):
                switches.append((det.previous_id, det.track_id, det))
        return switches

    def _create_pending_capture(
        self,
        old_id: int,
        new_id: int,
        switch_frame: FrameData,
        detection: "Detection",
    ) -> None:
        """Create pending capture for an ID switch event.

        Args:
            old_id: Previous ID before switch
            new_id: New ID after switch
            switch_frame: Frame where switch occurred
            detection: Detection object that switched
        """
        # Copy frames from buffer as before-frames
        frames_before = list(self._frame_buffer)

        pending = PendingCapture(
            old_id=old_id,
            new_id=new_id,
            switch_frame_idx=switch_frame.frame_idx,
            frames_before=frames_before,
            switch_frame=switch_frame,
            switch_bbox=detection.bbox,
        )
        self._pending_captures.append(pending)
        self._switch_count += 1

    def _update_pending_captures(self, frame_data: FrameData) -> None:
        """Add frame to pending captures and save completed ones.

        Args:
            frame_data: Current frame data
        """
        completed = []

        for pending in self._pending_captures:
            # Skip if this is the switch frame itself
            if frame_data.frame_idx == pending.switch_frame_idx:
                continue

            # Add as after-frame
            if len(pending.frames_after) < self._frames_after:
                pending.frames_after.append(frame_data)

            # Check if complete
            if len(pending.frames_after) >= self._frames_after:
                completed.append(pending)

        # Save and remove completed captures
        for pending in completed:
            self._save_capture(pending)
            self._pending_captures.remove(pending)

    def flush(self) -> None:
        """Save any remaining pending captures (called at end of video)."""
        for pending in self._pending_captures:
            # Mark as incomplete if missing after-frames
            is_incomplete = len(pending.frames_after) < self._frames_after
            self._save_capture(pending, is_incomplete=is_incomplete)
        self._pending_captures.clear()

    def _save_capture(self, capture: PendingCapture, is_incomplete: bool = False) -> None:
        """Save captured frames to disk.

        Args:
            capture: Completed PendingCapture with all frames
            is_incomplete: If True, capture has fewer after-frames than expected
        """
        # Create output directory
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Add incomplete indicator to filename prefix if applicable
        incomplete_tag = "_INCOMPLETE" if is_incomplete else ""
        prefix = f"{capture.old_id:03d}_to_{capture.new_id:03d}_frame_{capture.switch_frame_idx:06d}{incomplete_tag}"

        # Save before frames
        for i, frame_data in enumerate(capture.frames_before):
            filename = f"{prefix}_before_{i + 1}.jpg"
            annotated = self._annotate_frame(
                frame_data.frame,
                frame_data.detections,
                highlight_ids={capture.old_id, capture.new_id},
            )
            cv2.imwrite(str(self._output_dir / filename), annotated)

        # Save switch frame (full scene and crop)
        if capture.switch_frame is not None:
            # Full scene
            scene_filename = f"{prefix}_scene.jpg"
            annotated_scene = self._annotate_frame(
                capture.switch_frame.frame,
                capture.switch_frame.detections,
                highlight_ids={capture.old_id, capture.new_id},
                is_switch_frame=True,
            )
            cv2.imwrite(str(self._output_dir / scene_filename), annotated_scene)

            # Object crop
            if capture.switch_bbox is not None:
                crop_filename = f"{prefix}_crop.jpg"
                crop = self._extract_crop(capture.switch_frame.frame, capture.switch_bbox)
                cv2.imwrite(str(self._output_dir / crop_filename), crop)

        # Save after frames
        for i, frame_data in enumerate(capture.frames_after):
            filename = f"{prefix}_after_{i + 1}.jpg"
            annotated = self._annotate_frame(
                frame_data.frame,
                frame_data.detections,
                highlight_ids={capture.old_id, capture.new_id},
            )
            cv2.imwrite(str(self._output_dir / filename), annotated)

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: list["Detection"],
        highlight_ids: set[int] | None = None,
        is_switch_frame: bool = False,
    ) -> np.ndarray:
        """Annotate frame with bboxes and labels.

        Args:
            frame: BGR image
            detections: List of Detection objects
            highlight_ids: IDs to highlight with thicker boxes
            is_switch_frame: If True, add "SWITCH" label

        Returns:
            Annotated frame copy
        """
        vis = frame.copy()

        for det in detections:
            if det.track_id is None:
                continue

            x1, y1, x2, y2 = map(int, det.bbox)
            color = get_id_color(det.track_id) if det.track_id >= 0 else (128, 128, 128)

            # Thicker box for highlighted IDs
            thickness = 4 if highlight_ids and det.track_id in highlight_ids else 2
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

            # ID label
            label = f"ID:{det.track_id:02d}"
            if det.previous_id is not None and is_switch_frame:
                label = f"ID:{det.previous_id:02d}->ID:{det.track_id:02d}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
            cv2.putText(
                vis, label, (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

        # Add switch indicator
        if is_switch_frame:
            cv2.putText(
                vis, "ID SWITCH", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
            )

        return vis

    def _extract_crop(
        self,
        frame: np.ndarray,
        bbox: tuple[float, ...],
        padding: int = 20,
    ) -> np.ndarray:
        """Extract padded crop from frame.

        Args:
            frame: BGR image
            bbox: (x1, y1, x2, y2)
            padding: Pixel padding around bbox

        Returns:
            Cropped region
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        # Add padding with bounds check
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        return frame[y1:y2, x1:x2].copy()

    @property
    def switch_count(self) -> int:
        """Number of ID switches captured."""
        return self._switch_count

    @property
    def enabled(self) -> bool:
        """Whether capturer is enabled."""
        return self._enabled
