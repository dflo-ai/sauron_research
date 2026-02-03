"""HUD renderer for pipeline stages, statistics, and event ticker."""
from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np

from ..config import VisualizationConfig


class ReIDEvent(NamedTuple):
    """ReID event for ticker display."""

    message: str
    timestamp: float  # Frame index when event occurred
    event_type: str  # "match", "new", "lost"


class EventTicker:
    """Manages and renders recent ReID events."""

    def __init__(self, max_events: int = 3, fade_frames: int = 60):
        """Initialize event ticker.

        Args:
            max_events: Maximum events to display
            fade_frames: Frames before event fades out
        """
        self._events: deque[ReIDEvent] = deque(maxlen=max_events)
        self.fade_frames = fade_frames

    def add_event(self, message: str, frame_idx: int, event_type: str = "match") -> None:
        """Add event to ticker.

        Args:
            message: Event message text
            frame_idx: Current frame index
            event_type: Type of event (match, new, lost)
        """
        self._events.append(ReIDEvent(message, frame_idx, event_type))

    def render(
        self,
        frame_width: int,
        current_frame: int,
        height: int = 30,
    ) -> np.ndarray:
        """Render event ticker bar.

        Args:
            frame_width: Width of main frame
            current_frame: Current frame index for fade calculation
            height: Height of ticker bar

        Returns:
            BGR image of ticker bar
        """
        ticker = np.zeros((height, frame_width, 3), dtype=np.uint8)
        ticker[:] = (25, 25, 25)  # Dark background

        if not self._events:
            return ticker

        # Filter out old events and render active ones
        x_offset = 10
        for event in self._events:
            age = current_frame - event.timestamp
            if age > self.fade_frames:
                continue

            # Calculate opacity based on age
            opacity = max(0.0, 1.0 - (age / self.fade_frames))

            # Event type colors
            color_map = {
                "match": (0, 200, 100),  # Green
                "new": (200, 180, 0),  # Cyan-ish
                "lost": (80, 80, 200),  # Red-ish
            }
            color = color_map.get(event.event_type, (180, 180, 180))

            # Apply opacity to color
            color = tuple(int(c * opacity) for c in color)

            # Draw event text
            text = f"[{event.message}]"
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

            if x_offset + tw + 10 > frame_width:
                break  # No more space

            cv2.putText(
                ticker,
                text,
                (x_offset, height - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )
            x_offset += tw + 15

        return ticker


class HUDRenderer:
    """Renders pipeline HUD with stage indicators and stats."""

    STAGES = ["Detect", "Extract", "Match", "Track"]
    STAGE_ICONS = ["D", "E", "M", "T"]  # Simple letter icons

    def __init__(self, config: VisualizationConfig):
        """Initialize HUD renderer.

        Args:
            config: VisualizationConfig instance
        """
        self.height = 45
        self.bg_opacity = config.panel_bg_opacity
        self.event_ticker = EventTicker(max_events=3, fade_frames=60)

    def render_pipeline_bar(
        self,
        frame_width: int,
        active_stage: int,
    ) -> np.ndarray:
        """Render top HUD bar with pipeline stage indicators.

        Args:
            frame_width: Width of main frame
            active_stage: Index of currently active stage (0-3)

        Returns:
            BGR image of HUD bar
        """
        bar = np.zeros((self.height, frame_width, 3), dtype=np.uint8)
        bar[:] = (30, 30, 30)  # Dark background

        # Calculate stage layout
        stage_width = 100
        total_width = len(self.STAGES) * stage_width
        x_start = (frame_width - total_width) // 2

        for i, (stage, icon) in enumerate(zip(self.STAGES, self.STAGE_ICONS)):
            x = x_start + i * stage_width
            is_active = i == active_stage

            # Stage background
            if is_active:
                # Active stage: bright green background
                cv2.rectangle(bar, (x + 2, 5), (x + stage_width - 2, self.height - 5), (50, 180, 50), -1)
                text_color = (255, 255, 255)
            else:
                # Inactive: dark gray
                cv2.rectangle(bar, (x + 2, 5), (x + stage_width - 2, self.height - 5), (50, 50, 50), -1)
                text_color = (150, 150, 150)

            # Stage icon circle
            icon_x = x + 20
            icon_y = self.height // 2
            cv2.circle(bar, (icon_x, icon_y), 12, text_color, 2 if is_active else 1)
            cv2.putText(
                bar, icon, (icon_x - 5, icon_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1
            )

            # Stage label
            cv2.putText(
                bar, stage, (x + 38, icon_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1
            )

            # Arrow between stages
            if i < len(self.STAGES) - 1:
                arrow_x = x + stage_width - 5
                cv2.arrowedLine(
                    bar, (arrow_x, icon_y), (arrow_x + 8, icon_y),
                    (100, 100, 100), 1, tipLength=0.5
                )

        return bar

    def render_stats_panel(
        self,
        fps: float,
        frame_idx: int,
        total_frames: int,
        detections: int,
        reid_matches: int,
        unique_ids: int,
        threshold: float,
    ) -> np.ndarray:
        """Render statistics panel.

        Args:
            fps: Current FPS
            frame_idx: Current frame index
            total_frames: Total frames in video
            detections: Detections in current frame
            reid_matches: ReID matches in current frame
            unique_ids: Total unique IDs so far
            threshold: Current similarity threshold

        Returns:
            BGR image of stats panel
        """
        width = 180
        height = 110
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)

        # Draw border
        cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (60, 60, 60), 1)

        # Header
        cv2.putText(panel, "Stats", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.line(panel, (5, 24), (width - 5, 24), (60, 60, 60), 1)

        # Stats with visual hierarchy
        y = 42
        line_height = 18

        # Primary metrics (larger)
        cv2.putText(panel, f"IDs: {unique_ids}", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1)
        cv2.putText(panel, f"Det: {detections}", (95, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        y += line_height

        # ReID matches highlighted
        match_color = (0, 255, 100) if reid_matches > 0 else (150, 150, 150)
        cv2.putText(panel, f"ReID: {reid_matches}", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, match_color, 1)
        y += line_height

        # Frame progress
        progress = frame_idx / total_frames if total_frames > 0 else 0
        cv2.putText(panel, f"Frame: {frame_idx}/{total_frames}", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        y += line_height

        # Progress bar
        bar_width = width - 16
        cv2.rectangle(panel, (8, y - 2), (8 + bar_width, y + 6), (50, 50, 50), -1)
        cv2.rectangle(panel, (8, y - 2), (8 + int(bar_width * progress), y + 6), (80, 150, 80), -1)

        return panel

    def add_event(self, message: str, frame_idx: int, event_type: str = "match") -> None:
        """Add event to ticker."""
        self.event_ticker.add_event(message, frame_idx, event_type)

    def render_event_ticker(self, frame_width: int, current_frame: int) -> np.ndarray:
        """Render event ticker bar."""
        return self.event_ticker.render(frame_width, current_frame)

    def composite_hud(
        self,
        frame: np.ndarray,
        pipeline_bar: np.ndarray,
        stats_panel: np.ndarray,
        event_ticker: np.ndarray | None = None,
    ) -> np.ndarray:
        """Composite all HUD elements onto frame.

        Args:
            frame: Main video frame
            pipeline_bar: Top pipeline stage bar
            stats_panel: Stats panel (top-left)
            event_ticker: Optional bottom event ticker

        Returns:
            Frame with HUD overlaid
        """
        result = frame.copy()
        h, w = frame.shape[:2]

        # Pipeline bar at top (semi-transparent blend)
        bar_h = pipeline_bar.shape[0]
        roi = result[:bar_h, :]
        result[:bar_h, :] = cv2.addWeighted(pipeline_bar, self.bg_opacity, roi, 1 - self.bg_opacity, 0)

        # Stats panel at top-left (below pipeline bar)
        panel_h, panel_w = stats_panel.shape[:2]
        y_start = bar_h + 10
        roi = result[y_start : y_start + panel_h, 10 : 10 + panel_w]
        result[y_start : y_start + panel_h, 10 : 10 + panel_w] = cv2.addWeighted(
            stats_panel, self.bg_opacity, roi, 1 - self.bg_opacity, 0
        )

        # Event ticker at bottom
        if event_ticker is not None:
            ticker_h = event_ticker.shape[0]
            roi = result[h - ticker_h :, :]
            result[h - ticker_h :, :] = cv2.addWeighted(
                event_ticker, self.bg_opacity, roi, 1 - self.bg_opacity, 0
            )

        return result
