"""Tests for embedding extraction module."""

import pytest
import torch
from src.embeddings.extractor import EmbeddingExtractor, create_extractor


class TestEmbeddingExtractor:
    """Test suite for EmbeddingExtractor."""

    def test_extract_single_crop(self):
        """Test extracting embedding from single crop."""
        ext = EmbeddingExtractor(device="cpu", embedding_dim=256)
        crop = torch.randn(3, 256, 128)  # CHW

        emb = ext.extract([crop])

        assert emb.shape == (1, 256)

    def test_extract_normalized(self):
        """Test that embeddings are L2 normalized."""
        ext = EmbeddingExtractor(device="cpu", embedding_dim=256, normalize=True)
        crop = torch.randn(3, 256, 128)

        emb = ext.extract([crop])
        norm = emb.norm(dim=1)

        assert torch.allclose(norm, torch.ones(1), atol=1e-5)

    def test_extract_unnormalized(self):
        """Test embeddings without normalization."""
        ext = EmbeddingExtractor(device="cpu", embedding_dim=128, normalize=False)
        crop = torch.randn(3, 256, 128)

        emb = ext.extract([crop])
        norm = emb.norm(dim=1)

        # Should not necessarily be unit norm
        assert emb.shape == (1, 128)

    def test_extract_batch(self):
        """Test batch extraction."""
        ext = EmbeddingExtractor(device="cpu", embedding_dim=256)
        crops = torch.randn(5, 3, 256, 128)  # NCHW batch

        emb = ext.extract(crops)

        assert emb.shape == (5, 256)

    def test_extract_list_of_crops(self):
        """Test extraction from list of variable size crops."""
        ext = EmbeddingExtractor(device="cpu", embedding_dim=256)
        crops = [
            torch.randn(3, 200, 100),
            torch.randn(3, 300, 150),
            torch.randn(3, 150, 80),
        ]

        emb = ext.extract(crops)

        assert emb.shape == (3, 256)

    def test_extract_empty_list(self):
        """Test extraction from empty list."""
        ext = EmbeddingExtractor(device="cpu", embedding_dim=256)

        emb = ext.extract([])

        assert emb.shape == (0, 256)

    def test_extract_from_frame(self):
        """Test extracting embeddings from frame with boxes."""
        ext = EmbeddingExtractor(device="cpu", embedding_dim=256)
        frame = torch.randn(480, 640, 3)  # HWC format
        boxes = torch.tensor([
            [10, 10, 100, 200],
            [200, 50, 300, 250],
        ], dtype=torch.float)

        emb = ext.extract_from_frame(frame, boxes)

        assert emb.shape == (2, 256)

    def test_extract_from_frame_chw(self):
        """Test extraction from CHW format frame."""
        ext = EmbeddingExtractor(device="cpu", embedding_dim=256)
        frame = torch.randn(3, 480, 640)  # CHW format
        boxes = torch.tensor([[50, 50, 150, 200]], dtype=torch.float)

        emb = ext.extract_from_frame(frame, boxes)

        assert emb.shape == (1, 256)

    def test_extract_from_frame_empty_boxes(self):
        """Test extraction with no boxes."""
        ext = EmbeddingExtractor(device="cpu", embedding_dim=256)
        frame = torch.randn(3, 480, 640)

        emb = ext.extract_from_frame(frame, torch.empty(0, 4))

        assert emb.shape == (0, 256)

    def test_skip_tiny_boxes(self):
        """Test that tiny boxes are skipped (return zeros)."""
        ext = EmbeddingExtractor(device="cpu", embedding_dim=256)
        frame = torch.randn(3, 480, 640)
        boxes = torch.tensor([
            [10, 10, 100, 200],  # Valid: 90x190
            [200, 200, 220, 220],  # Tiny: 20x20
        ], dtype=torch.float)

        emb = ext.extract_from_frame(frame, boxes, min_size=50)

        assert emb.shape == (2, 256)
        # First should have non-zero embedding
        assert emb[0].abs().sum() > 0
        # Second should be zeros (skipped)
        assert emb[1].abs().sum() == 0

    def test_different_models(self):
        """Test different backbone models."""
        for model_name in ["resnet18", "resnet50"]:
            ext = EmbeddingExtractor(
                device="cpu",
                model_name=model_name,
                embedding_dim=128,
            )
            crop = torch.randn(3, 256, 128)
            emb = ext.extract([crop])

            assert emb.shape == (1, 128)

    def test_values_in_255_range(self):
        """Test handling of [0, 255] input range."""
        ext = EmbeddingExtractor(device="cpu", embedding_dim=256)
        crop = torch.randint(0, 256, (3, 256, 128), dtype=torch.float)

        emb = ext.extract([crop])

        assert emb.shape == (1, 256)
        # Should still be normalized
        assert torch.allclose(emb.norm(dim=1), torch.ones(1), atol=1e-5)

    def test_create_extractor_factory(self):
        """Test factory function."""
        ext = create_extractor(model="resnet18", dim=128, device="cpu")

        assert ext.embedding_dim == 128
        assert isinstance(ext, EmbeddingExtractor)

    def test_invalid_model_raises(self):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError, match="Unsupported model"):
            EmbeddingExtractor(device="cpu", model_name="invalid_model")

    def test_box_clamping_to_frame(self):
        """Test that boxes outside frame are clamped."""
        ext = EmbeddingExtractor(device="cpu", embedding_dim=256)
        frame = torch.randn(3, 100, 100)
        # Box extends beyond frame boundaries
        boxes = torch.tensor([[-10, -10, 50, 50]], dtype=torch.float)

        # Should not raise, should clamp to valid region
        emb = ext.extract_from_frame(frame, boxes)
        assert emb.shape == (1, 256)
