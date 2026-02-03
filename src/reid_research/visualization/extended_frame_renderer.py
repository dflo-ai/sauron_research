"""Extended frame renderer with analytics outside main video area."""
import cv2
import numpy as np

from ..config import VisualizationConfig
from .colors import get_id_color
from .gallery_panel import GalleryPanelEntry


class ExtendedFrameRenderer:
    """Renders video with extended padding for analytics outside frame."""

    def __init__(self, config: VisualizationConfig):
        """Initialize renderer.

        Args:
            config: VisualizationConfig instance
        """
        self.config = config
        self.left_panel_width = 180  # Stats panel
        self.right_panel_width = config.gallery_panel_width  # Gallery panel
        self.bottom_bar_height = 40  # ID matching bar (replaces top HUD)
        self.bg_color = (25, 25, 25)  # Dark background

    def create_extended_frame(
        self,
        video_frame: np.ndarray,
        gallery_entries: list[GalleryPanelEntry],
        stats: dict,
        recent_matches: list[dict],
        fps: int = 30,
    ) -> np.ndarray:
        """Create extended frame with analytics in padding areas.

        Args:
            video_frame: Main video frame with bbox annotations
            gallery_entries: Gallery entries for right panel
            stats: Dict with frame_idx, total_frames, detections, reid_matches, unique_ids
            recent_matches: List of recent ID matches [{id, similarity, is_new}, ...]
            fps: Video FPS

        Returns:
            Extended frame with analytics panels
        """
        vh, vw = video_frame.shape[:2]

        # Calculate extended dimensions
        total_w = self.left_panel_width + vw + self.right_panel_width
        total_h = vh + self.bottom_bar_height

        # Create canvas
        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        canvas[:] = self.bg_color

        # Place video frame in center
        x_offset = self.left_panel_width
        canvas[:vh, x_offset:x_offset + vw] = video_frame

        # Draw left stats panel
        self._draw_stats_panel(canvas, stats, vh)

        # Draw right gallery panel
        self._draw_gallery_panel(canvas, gallery_entries, vh, vw, fps)

        # Draw bottom ID matching bar (replaces top pipeline HUD)
        self._draw_id_matching_bar(canvas, recent_matches, vh, total_w)

        return canvas

    def _draw_stats_panel(
        self, canvas: np.ndarray, stats: dict, video_height: int
    ) -> None:
        """Draw stats panel on left side."""
        x, y = 10, 20
        line_h = 22

        # Title
        cv2.putText(canvas, "Stats", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 8
        cv2.line(canvas, (x, y), (self.left_panel_width - 10, y), (60, 60, 60), 1)
        y += line_h

        # Frame progress
        frame_idx = stats.get("frame_idx", 0)
        total = stats.get("total_frames", 1)
        progress = frame_idx / total if total > 0 else 0

        cv2.putText(canvas, f"Frame: {frame_idx}/{total}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        y += line_h

        # Progress bar
        bar_w = self.left_panel_width - 20
        cv2.rectangle(canvas, (x, y - 5), (x + bar_w, y + 5), (50, 50, 50), -1)
        cv2.rectangle(canvas, (x, y - 5), (x + int(bar_w * progress), y + 5), (80, 150, 80), -1)
        y += line_h + 5

        # Key metrics
        unique_ids = stats.get("unique_ids", 0)
        detections = stats.get("detections", 0)
        reid_matches = stats.get("reid_matches", 0)

        cv2.putText(canvas, f"Unique IDs: {unique_ids}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        y += line_h

        cv2.putText(canvas, f"Detections: {detections}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        y += line_h

        match_color = (0, 255, 100) if reid_matches > 0 else (150, 150, 150)
        cv2.putText(canvas, f"ReID Matches: {reid_matches}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, match_color, 1)

    def _draw_gallery_panel(
        self,
        canvas: np.ndarray,
        entries: list[GalleryPanelEntry],
        video_height: int,
        video_width: int,
        fps: int,
    ) -> None:
        """Draw gallery panel on right side."""
        x_start = self.left_panel_width + video_width + 10
        y = 20

        # Title
        cv2.putText(canvas, "Gallery", (x_start, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 8
        cv2.line(canvas, (x_start, y), (x_start + self.right_panel_width - 20, y), (60, 60, 60), 1)
        y += 15

        # Render entries (max 8)
        thumb_size = (50, 65)
        entry_height = 75

        for entry in entries[:8]:
            if y + entry_height > video_height - 10:
                break

            color = get_id_color(entry.track_id)

            # Colored border for active
            border_thick = 3 if entry.is_active else 1
            cv2.rectangle(canvas, (x_start, y), (x_start + self.right_panel_width - 20, y + entry_height - 5), color, border_thick)

            # Thumbnail
            if entry.thumbnail is not None:
                thumb = cv2.resize(entry.thumbnail, thumb_size)
                canvas[y + 5:y + 5 + thumb_size[1], x_start + 5:x_start + 5 + thumb_size[0]] = thumb

            # ID and info
            text_x = x_start + thumb_size[0] + 10
            cv2.putText(canvas, f"ID:{entry.track_id:02d}", (text_x, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # First/last seen
            first_t = self._format_time(entry.first_seen, fps)
            last_t = self._format_time(entry.last_seen, fps)
            cv2.putText(canvas, f"In:{first_t}", (text_x, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            cv2.putText(canvas, f"Out:{last_t}", (text_x, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

            # Active indicator
            if entry.is_active:
                cv2.circle(canvas, (x_start + self.right_panel_width - 30, y + 35), 5, (0, 255, 0), -1)

            y += entry_height

    def _draw_id_matching_bar(
        self,
        canvas: np.ndarray,
        recent_matches: list[dict],
        video_height: int,
        total_width: int,
    ) -> None:
        """Draw ID matching bar at bottom (replaces top pipeline HUD)."""
        y_start = video_height
        bar_height = self.bottom_bar_height

        # Background
        cv2.rectangle(canvas, (0, y_start), (total_width, y_start + bar_height), (35, 35, 35), -1)

        # Title
        cv2.putText(canvas, "ID Matching:", (10, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Recent matches
        x = 120
        for match in recent_matches[-5:]:  # Show last 5 matches
            track_id = match.get("id", 0)
            similarity = match.get("similarity", 0.0)
            is_new = match.get("is_new", False)

            color = get_id_color(track_id)

            if is_new:
                text = f"[ID:{track_id:02d} NEW]"
                text_color = (200, 180, 0)
            else:
                text = f"[ID:{track_id:02d} {similarity:.2f}]"
                text_color = (0, 200, 100)

            # Colored box background
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(canvas, (x - 2, y_start + 8), (x + tw + 4, y_start + 32), color, 2)
            cv2.putText(canvas, text, (x, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)

            x += tw + 15
            if x > total_width - 100:
                break

    def _format_time(self, frame_idx: int, fps: int) -> str:
        """Format frame index as MM:SS."""
        if fps <= 0:
            fps = 30
        total_seconds = int(frame_idx / fps)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    def get_extended_dimensions(self, video_width: int, video_height: int) -> tuple[int, int]:
        """Get dimensions for extended frame.

        Args:
            video_width: Original video width
            video_height: Original video height

        Returns:
            (extended_width, extended_height)
        """
        w = self.left_panel_width + video_width + self.right_panel_width
        h = video_height + self.bottom_bar_height
        return (w, h)
