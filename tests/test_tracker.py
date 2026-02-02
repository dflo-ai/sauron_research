"""Tests for HAT-enhanced tracker module."""

import pytest
import torch
from src.tracker.hat_tracker import HATTracker, Track


class TestHATTracker:
    """Test suite for HATTracker."""

    def test_new_tracks_created(self):
        """Test that new tracks are created for high-score detections."""
        tracker = HATTracker(device="cpu", init_score_thr=0.8)
        boxes = torch.tensor([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=torch.float)
        scores = torch.tensor([0.9, 0.85])
        embeds = torch.randn(2, 256)

        ids = tracker.update(boxes, scores, embeds, frame_id=0)

        assert (ids >= 0).all()  # Both should get IDs
        assert tracker.num_tracks == 2

    def test_low_score_rejected(self):
        """Test that low-score detections don't create tracks."""
        tracker = HATTracker(device="cpu", init_score_thr=0.8)
        boxes = torch.tensor([[10, 10, 50, 50]], dtype=torch.float)
        scores = torch.tensor([0.5])  # Below threshold
        embeds = torch.randn(1, 256)

        ids = tracker.update(boxes, scores, embeds, frame_id=0)

        assert ids[0] == -1
        assert tracker.num_tracks == 0

    def test_track_association(self):
        """Test that similar embeddings associate to same track."""
        tracker = HATTracker(device="cpu", match_score_thr=0.3)

        # Frame 0: create track with specific embedding
        embed0 = torch.randn(1, 256)
        embed0 = torch.nn.functional.normalize(embed0, dim=1)
        tracker.update(
            torch.tensor([[10, 10, 50, 50]], dtype=torch.float),
            torch.tensor([0.9]),
            embed0,
            frame_id=0
        )

        # Frame 1: use similar embedding (should match)
        embed1 = embed0 + torch.randn(1, 256) * 0.1  # Small noise
        embed1 = torch.nn.functional.normalize(embed1, dim=1)
        ids = tracker.update(
            torch.tensor([[12, 12, 52, 52]], dtype=torch.float),
            torch.tensor([0.9]),
            embed1,
            frame_id=1
        )

        assert ids[0] == 0  # Should match first track

    def test_lost_track_cleanup(self):
        """Test that lost tracks are removed after max_lost_frames."""
        tracker = HATTracker(device="cpu", max_lost_frames=3)

        # Create track
        tracker.update(
            torch.tensor([[10, 10, 50, 50]], dtype=torch.float),
            torch.tensor([0.9]),
            torch.randn(1, 256),
            frame_id=0
        )
        assert tracker.num_tracks == 1

        # Simulate frames without detection
        for f in range(1, 5):
            tracker.update(
                torch.empty(0, 4),
                torch.empty(0),
                torch.empty(0, 256),
                frame_id=f
            )

        # Track should be removed after 3 frames
        assert tracker.num_tracks == 0

    def test_empty_detections(self):
        """Test handling of empty detection input."""
        tracker = HATTracker(device="cpu")

        ids = tracker.update(
            torch.empty(0, 4),
            torch.empty(0),
            torch.empty(0, 256),
            frame_id=0
        )

        assert len(ids) == 0
        assert tracker.num_tracks == 0

    def test_hat_activation_threshold(self):
        """Test HAT activates only when history exceeds threshold."""
        tracker = HATTracker(
            device="cpu",
            use_hat=True,
            hat_factor_thr=4.0,
            history_max_len=100,
            match_score_thr=0.1,  # Low threshold to ensure matching
        )

        # Use consistent embeddings so tracks associate properly
        base_embeds = [
            torch.nn.functional.normalize(torch.randn(1, 256), dim=1),
            torch.nn.functional.normalize(torch.randn(1, 256), dim=1),
            torch.nn.functional.normalize(torch.randn(1, 256), dim=1),
        ]

        # Create multiple tracks with history
        for frame in range(20):
            boxes = torch.tensor([
                [10, 10, 50, 50],
                [100, 100, 150, 150],
                [200, 200, 250, 250],
            ], dtype=torch.float)
            scores = torch.tensor([0.9, 0.85, 0.88])
            # Add small noise to base embeddings
            embeds = torch.cat([
                base_embeds[0] + torch.randn(1, 256) * 0.05,
                base_embeds[1] + torch.randn(1, 256) * 0.05,
                base_embeds[2] + torch.randn(1, 256) * 0.05,
            ], dim=0)
            embeds = torch.nn.functional.normalize(embeds, dim=1)
            tracker.update(boxes, scores, embeds, frame_id=frame)

        # With 3 tracks and ~20 samples each = 60 history samples
        # HAT threshold: 4 * 3 = 12, so 60 > 12 should activate
        # Check total history
        total_hist = sum(len(t.history) for t in tracker.tracks.values())
        assert total_hist > tracker.hat_factor_thr * tracker.num_tracks
        assert tracker.hat_active

    def test_fifo_queue_type(self):
        """Test tracker with FIFO queue."""
        tracker = HATTracker(device="cpu", queue_type="fifo")

        tracker.update(
            torch.tensor([[10, 10, 50, 50]], dtype=torch.float),
            torch.tensor([0.9]),
            torch.randn(1, 256),
            frame_id=0
        )

        from src.hat_reid.queues import FIFOQueue
        track = list(tracker.tracks.values())[0]
        assert isinstance(track.history, FIFOQueue)

    def test_score_queue_type(self):
        """Test tracker with Score queue."""
        tracker = HATTracker(device="cpu", queue_type="score")

        tracker.update(
            torch.tensor([[10, 10, 50, 50]], dtype=torch.float),
            torch.tensor([0.9]),
            torch.randn(1, 256),
            frame_id=0
        )

        from src.hat_reid.queues import ScoreQueue
        track = list(tracker.tracks.values())[0]
        assert isinstance(track.history, ScoreQueue)

    def test_reset_clears_state(self):
        """Test that reset clears all tracks."""
        tracker = HATTracker(device="cpu")

        tracker.update(
            torch.tensor([[10, 10, 50, 50]], dtype=torch.float),
            torch.tensor([0.9]),
            torch.randn(1, 256),
            frame_id=0
        )
        assert tracker.num_tracks == 1

        tracker.reset()
        assert tracker.num_tracks == 0
        assert tracker.next_id == 0

    def test_embedding_momentum_update(self):
        """Test that track embeddings are updated with momentum."""
        tracker = HATTracker(device="cpu", memo_momentum=0.5, match_score_thr=0.1)

        # Use normalized embeddings for proper cosine similarity matching
        embed0 = torch.nn.functional.normalize(torch.ones(1, 64), dim=1)
        tracker.update(
            torch.tensor([[10, 10, 50, 50]], dtype=torch.float),
            torch.tensor([0.9]),
            embed0,
            frame_id=0
        )

        old_embed = tracker.tracks[0].embed.clone()

        # Update with similar embedding to ensure matching (normalized ones + small noise)
        embed1 = torch.nn.functional.normalize(torch.ones(1, 64) * 0.9, dim=1)
        tracker.update(
            torch.tensor([[10, 10, 50, 50]], dtype=torch.float),
            torch.tensor([0.9]),
            embed1,
            frame_id=1
        )

        new_embed = tracker.tracks[0].embed
        # With momentum=0.5: new = 0.5*old + 0.5*new_input
        expected = 0.5 * old_embed + 0.5 * embed1.squeeze()
        assert torch.allclose(new_embed, expected, atol=1e-5)

    def test_multiple_tracks_different_ids(self):
        """Test that multiple detections get unique track IDs."""
        tracker = HATTracker(device="cpu")

        boxes = torch.tensor([
            [10, 10, 50, 50],
            [100, 100, 150, 150],
            [200, 200, 250, 250],
        ], dtype=torch.float)
        scores = torch.tensor([0.9, 0.85, 0.88])
        embeds = torch.randn(3, 256)

        ids = tracker.update(boxes, scores, embeds, frame_id=0)

        assert len(set(ids.tolist())) == 3  # All unique IDs
        assert tracker.num_tracks == 3

    def test_similarity_modes(self):
        """Test different similarity computation modes."""
        for mode in ["cosine", "bisoftmax", "masa"]:
            tracker = HATTracker(device="cpu", similarity_mode=mode)

            # Create initial tracks
            tracker.update(
                torch.tensor([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=torch.float),
                torch.tensor([0.9, 0.85]),
                torch.randn(2, 256),
                frame_id=0
            )

            # Update should work without error
            ids = tracker.update(
                torch.tensor([[12, 12, 52, 52]], dtype=torch.float),
                torch.tensor([0.9]),
                torch.randn(1, 256),
                frame_id=1
            )

            assert len(ids) == 1
