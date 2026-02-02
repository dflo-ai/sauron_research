"""Tests for LDA transformation module."""

import pytest
import torch
from src.hat_reid.lda import LDA


class TestLDA:
    """Test suite for LDA class."""

    def test_fit_transform_basic(self):
        """Test basic fit and transform with multiple classes."""
        lda = LDA(device="cpu")
        X = torch.randn(100, 256)  # 100 samples, 256-dim
        y = torch.randint(0, 5, (100,))  # 5 tracks

        lda.fit(X, y)
        X_t = lda.transform(X)

        assert lda.is_fitted()
        assert X_t.shape == (100, 4)  # 5 classes -> 4 dims

    def test_fit_with_scores(self):
        """Test fit with detection scores as weights."""
        lda = LDA(use_weighted_class_mean=True, device="cpu")
        X = torch.randn(50, 128)
        y = torch.randint(0, 3, (50,))
        scores = torch.rand(50) * 0.5 + 0.5  # Scores in [0.5, 1.0]

        lda.fit(X, y, scores=scores)
        X_t = lda.transform(X)

        assert lda.is_fitted()
        assert X_t.shape == (50, 2)  # 3 classes -> 2 dims

    def test_single_class_no_fit(self):
        """Test that LDA doesn't fit with only one class."""
        lda = LDA(device="cpu")
        X = torch.randn(20, 64)
        y = torch.zeros(20, dtype=torch.long)  # All same class

        lda.fit(X, y)

        assert not lda.is_fitted()
        # Transform should return unchanged
        X_t = lda.transform(X)
        assert torch.allclose(X_t, X.to(lda.dtype))

    def test_two_classes(self):
        """Test with exactly two classes (minimum for LDA)."""
        lda = LDA(device="cpu")
        X = torch.randn(40, 32)
        y = torch.tensor([0] * 20 + [1] * 20)

        lda.fit(X, y)
        X_t = lda.transform(X)

        assert lda.is_fitted()
        assert X_t.shape == (40, 1)  # 2 classes -> 1 dim

    def test_shrinkage_prevents_singular(self):
        """Test that shrinkage prevents singular matrix errors."""
        lda = LDA(use_shrinkage=True, device="cpu")
        # High-dimensional with few samples per class (prone to singularity)
        X = torch.randn(30, 256)
        y = torch.randint(0, 10, (30,))  # 10 classes, ~3 samples each

        # Should not raise error due to shrinkage
        lda.fit(X, y)
        if lda.is_fitted():
            X_t = lda.transform(X)
            assert X_t.shape[0] == 30

    def test_fit_transform_combined(self):
        """Test fit_transform method."""
        lda = LDA(device="cpu")
        X = torch.randn(60, 64)
        y = torch.randint(0, 4, (60,))

        X_t = lda.fit_transform(X, y)

        assert lda.is_fitted()
        assert X_t.shape == (60, 3)

    def test_clear_resets_state(self):
        """Test that clear() resets fitted state."""
        lda = LDA(device="cpu")
        X = torch.randn(50, 64)
        y = torch.randint(0, 3, (50,))

        lda.fit(X, y)
        assert lda.is_fitted()

        lda.clear()
        assert not lda.is_fitted()
        assert lda.classes is None
        assert lda.project_matrix is None

    def test_transform_single_vector(self):
        """Test transform on single feature vector."""
        lda = LDA(device="cpu")
        X = torch.randn(50, 64)
        y = torch.randint(0, 3, (50,))

        lda.fit(X, y)
        single = torch.randn(64)
        result = lda.transform(single)

        assert result.shape == (2,)  # 3 classes -> 2 dims

    def test_dtype_consistency(self):
        """Test that output dtype matches config."""
        lda = LDA(dtype=torch.float64, device="cpu")
        X = torch.randn(50, 64, dtype=torch.float32)
        y = torch.randint(0, 3, (50,))

        lda.fit(X, y)
        X_t = lda.transform(X)

        assert X_t.dtype == torch.float64
