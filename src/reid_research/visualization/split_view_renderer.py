"""Split-view renderer for showing multiple pipeline stages side-by-side."""
import cv2
import numpy as np

from ..config import VisualizationConfig


class SplitViewRenderer:
    """Renders split-view layout showing pipeline stages side-by-side."""

    def __init__(self, config: VisualizationConfig):
        """Initialize split view renderer.

        Args:
            config: VisualizationConfig instance
        """
        self.enabled = config.split_view_enabled
        self.stages = config.split_view_stages
        self.panel_border = 2
        self.label_height = 25

    def render(
        self,
        frames: dict[str, np.ndarray],
        output_size: tuple[int, int],
    ) -> np.ndarray:
        """Combine stage frames into split view.

        Args:
            frames: Dict mapping stage name to frame (e.g., {"detection": frame, "reid": frame})
            output_size: Target output size (width, height)

        Returns:
            Combined split-view frame
        """
        if not self.enabled:
            # Return first available frame if not enabled
            return next(iter(frames.values()))

        output_w, output_h = output_size
        num_stages = len(self.stages)

        # Filter to requested stages
        stage_frames = []
        stage_labels = []
        for stage in self.stages:
            if stage in frames:
                stage_frames.append(frames[stage])
                stage_labels.append(stage.capitalize())

        if not stage_frames:
            # Fallback: return first available frame
            return next(iter(frames.values()))

        # Choose layout based on number of stages
        if len(stage_frames) == 2:
            return self._layout_2way(stage_frames, stage_labels, output_w, output_h)
        elif len(stage_frames) == 4:
            return self._layout_4way(stage_frames, stage_labels, output_w, output_h)
        else:
            # Default to 2-way with first and last stage
            return self._layout_2way(
                [stage_frames[0], stage_frames[-1]],
                [stage_labels[0], stage_labels[-1]],
                output_w, output_h
            )

    def _layout_2way(
        self,
        panels: list[np.ndarray],
        labels: list[str],
        output_w: int,
        output_h: int,
    ) -> np.ndarray:
        """Side-by-side layout (2 panels).

        Args:
            panels: List of 2 frames
            labels: List of 2 labels
            output_w: Output width
            output_h: Output height

        Returns:
            Combined frame
        """
        panel_w = (output_w - self.panel_border * 2) // 2
        panel_h = output_h

        # Resize panels
        left = self._resize_panel(panels[0], (panel_w, panel_h))
        right = self._resize_panel(panels[1], (panel_w, panel_h))

        # Add labels
        left = self._add_panel_label(left, labels[0])
        right = self._add_panel_label(right, labels[1])

        # Create border
        border = np.ones((panel_h, self.panel_border * 2, 3), dtype=np.uint8) * 80

        # Combine
        return np.hstack([left, border, right])

    def _layout_4way(
        self,
        panels: list[np.ndarray],
        labels: list[str],
        output_w: int,
        output_h: int,
    ) -> np.ndarray:
        """2x2 grid layout (4 panels).

        Args:
            panels: List of 4 frames
            labels: List of 4 labels
            output_w: Output width
            output_h: Output height

        Returns:
            Combined frame
        """
        panel_w = (output_w - self.panel_border * 2) // 2
        panel_h = (output_h - self.panel_border * 2) // 2

        # Resize all panels
        resized = [self._resize_panel(p, (panel_w, panel_h)) for p in panels]

        # Add labels
        labeled = [self._add_panel_label(p, l) for p, l in zip(resized, labels)]

        # Create borders
        v_border = np.ones((panel_h, self.panel_border * 2, 3), dtype=np.uint8) * 80
        h_border = np.ones((self.panel_border * 2, output_w, 3), dtype=np.uint8) * 80

        # Combine
        top = np.hstack([labeled[0], v_border, labeled[1]])
        bottom = np.hstack([labeled[2], v_border, labeled[3]])
        return np.vstack([top, h_border, bottom])

    def _resize_panel(
        self, frame: np.ndarray, target_size: tuple[int, int]
    ) -> np.ndarray:
        """Resize frame to target size with letterboxing.

        Args:
            frame: Input frame
            target_size: Target (width, height)

        Returns:
            Resized frame with preserved aspect ratio
        """
        target_w, target_h = target_size
        h, w = frame.shape[:2]

        # Calculate scale to fit
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create letterboxed output
        result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        result[:] = (30, 30, 30)  # Dark gray background

        # Center the resized frame
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        result[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        return result

    def _add_panel_label(self, panel: np.ndarray, label: str) -> np.ndarray:
        """Add stage label at top of panel.

        Args:
            panel: Panel image
            label: Label text

        Returns:
            Panel with label overlay
        """
        result = panel.copy()
        h, w = result.shape[:2]

        # Semi-transparent label bar
        overlay = result.copy()
        cv2.rectangle(overlay, (0, 0), (w, self.label_height), (0, 0, 0), -1)
        result = cv2.addWeighted(overlay, 0.7, result, 0.3, 0)

        # Label text
        cv2.putText(
            result, label, (10, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )

        return result
