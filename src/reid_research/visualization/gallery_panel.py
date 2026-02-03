"""Gallery side panel renderer showing tracked person thumbnails."""
from dataclasses import dataclass

import cv2
import numpy as np

from ..config import VisualizationConfig
from .colors import get_id_color


@dataclass
class GalleryPanelEntry:
    """Data for a single gallery panel entry."""

    track_id: int
    thumbnail: np.ndarray  # Pre-resized BGR image
    first_seen: int  # Frame index
    last_seen: int  # Frame index
    detection_count: int
    is_active: bool = False  # Currently visible in frame


class GalleryPanelRenderer:
    """Renders gallery side panel with person thumbnails."""

    def __init__(self, config: VisualizationConfig):
        """Initialize renderer with config.

        Args:
            config: VisualizationConfig instance
        """
        self.width = config.gallery_panel_width
        self.max_entries = config.max_gallery_entries
        self.bg_opacity = config.panel_bg_opacity
        self.position = config.gallery_panel_position

        # Layout constants
        self.thumbnail_size = (60, 80)  # W, H
        self.entry_height = 90
        self.padding = 8
        self.header_height = 30

    def render(
        self,
        frame_height: int,
        entries: list[GalleryPanelEntry],
        fps: int = 30,
    ) -> np.ndarray:
        """Render gallery panel as separate image.

        Args:
            frame_height: Height of main video frame
            entries: List of gallery entries (most recent first)
            fps: Video FPS for time formatting

        Returns:
            BGR panel image (frame_height x self.width)
        """
        # Create panel with semi-transparent dark background
        panel = np.zeros((frame_height, self.width, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)  # Dark gray background

        # Draw header
        cv2.putText(
            panel,
            "Gallery",
            (self.padding, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        cv2.line(
            panel,
            (self.padding, self.header_height),
            (self.width - self.padding, self.header_height),
            (80, 80, 80),
            1,
        )

        # Render entries (most recent first, limit to max_entries)
        display_entries = entries[: self.max_entries]
        y_offset = self.header_height + self.padding

        for entry in display_entries:
            if y_offset + self.entry_height > frame_height - self.padding:
                break  # No more space

            self._render_entry(panel, entry, y_offset, fps)
            y_offset += self.entry_height + self.padding

        return panel

    def _render_entry(
        self,
        panel: np.ndarray,
        entry: GalleryPanelEntry,
        y_offset: int,
        fps: int,
    ) -> None:
        """Render single gallery entry on panel.

        Args:
            panel: Panel image to draw on
            entry: Entry data to render
            y_offset: Y position for this entry
            fps: Video FPS for time formatting
        """
        color = get_id_color(entry.track_id)
        x_start = self.padding

        # Entry background with colored border
        entry_bg = (40, 40, 40) if not entry.is_active else (50, 50, 60)
        cv2.rectangle(
            panel,
            (x_start, y_offset),
            (self.width - self.padding, y_offset + self.entry_height),
            entry_bg,
            -1,
        )

        # Colored left border (thicker for active)
        border_thickness = 4 if entry.is_active else 2
        cv2.rectangle(
            panel,
            (x_start, y_offset),
            (x_start + border_thickness, y_offset + self.entry_height),
            color,
            -1,
        )

        # Active indicator glow
        if entry.is_active:
            cv2.rectangle(
                panel,
                (x_start, y_offset),
                (self.width - self.padding, y_offset + self.entry_height),
                color,
                2,
            )

        # Thumbnail
        thumb_x = x_start + border_thickness + 4
        thumb_y = y_offset + 5
        thumb_w, thumb_h = self.thumbnail_size

        if entry.thumbnail is not None:
            # Resize thumbnail if needed
            thumb = cv2.resize(entry.thumbnail, self.thumbnail_size)
            # Place thumbnail
            panel[thumb_y : thumb_y + thumb_h, thumb_x : thumb_x + thumb_w] = thumb

        # ID label
        id_text = f"ID:{entry.track_id:02d}"
        text_x = thumb_x + thumb_w + 6
        cv2.putText(
            panel,
            id_text,
            (text_x, y_offset + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

        # Time info (first seen / last seen)
        first_time = self._format_time(entry.first_seen, fps)
        last_time = self._format_time(entry.last_seen, fps)
        cv2.putText(
            panel,
            f"In: {first_time}",
            (text_x, y_offset + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (180, 180, 180),
            1,
        )
        cv2.putText(
            panel,
            f"Out: {last_time}",
            (text_x, y_offset + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (180, 180, 180),
            1,
        )

        # Detection count
        cv2.putText(
            panel,
            f"#{entry.detection_count}",
            (text_x, y_offset + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (150, 150, 150),
            1,
        )

    def _format_time(self, frame_idx: int, fps: int) -> str:
        """Format frame index as MM:SS."""
        if fps <= 0:
            fps = 30
        total_seconds = int(frame_idx / fps)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    def composite_on_frame(
        self,
        frame: np.ndarray,
        panel: np.ndarray,
    ) -> np.ndarray:
        """Composite panel onto main frame.

        Args:
            frame: Main video frame
            panel: Rendered panel image

        Returns:
            Frame with panel overlaid
        """
        h, w = frame.shape[:2]
        panel_h, panel_w = panel.shape[:2]

        # Determine x position based on config
        if self.position == "left":
            x_start = 0
        else:  # "right"
            x_start = w - panel_w

        # Blend panel onto frame with opacity
        result = frame.copy()
        roi = result[:panel_h, x_start : x_start + panel_w]
        blended = cv2.addWeighted(panel, self.bg_opacity, roi, 1 - self.bg_opacity, 0)
        result[:panel_h, x_start : x_start + panel_w] = blended

        return result
