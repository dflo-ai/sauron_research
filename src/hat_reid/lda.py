"""History-Aware Transformation via Linear Discriminant Analysis.

Port of HAT-MASA LDA implementation with shrinkage covariance estimation.
Solves generalized eigenvalue problem: S_b @ v = lambda * S_w @ v
to find discriminative subspace maximizing between-class separation.
"""

import torch
import scipy.linalg
from torch import Tensor
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler


class LDA:
    """LDA-based feature transformation for ReID.

    Learns projection matrix from track history to maximize separation
    between different track identities. Uses shrinkage covariance for
    numerical stability with limited samples.
    """

    def __init__(
        self,
        use_shrinkage: bool = True,
        use_weighted_class_mean: bool = True,
        weighted_class_mean_alpha: float = 1.0,
        use_sample_average: bool = True,
        weight_min: float = 0.0,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cuda",
    ):
        """Initialize LDA transformer.

        Args:
            use_shrinkage: Use shrinkage covariance estimation (recommended)
            use_weighted_class_mean: Weight class means by detection scores
            weighted_class_mean_alpha: Exponent for score weighting
            use_sample_average: Weight covariance by sample counts
            weight_min: Minimum weight clamp value
            dtype: Tensor dtype for computation
            device: Device for computation
        """
        self.use_shrinkage = use_shrinkage
        self.use_weighted_class_mean = use_weighted_class_mean
        self.alpha = weighted_class_mean_alpha
        self.use_sample_average = use_sample_average
        self.weight_min = weight_min
        self.dtype = dtype
        self.device = torch.device(device) if isinstance(device, str) else device

        self.classes: list[int] | None = None
        self.class_means: Tensor | None = None
        self.project_matrix: Tensor | None = None

    def clear(self) -> None:
        """Reset fitted state."""
        self.classes = None
        self.class_means = None
        self.project_matrix = None

    def is_fitted(self) -> bool:
        """Check if LDA has been fitted."""
        return self.project_matrix is not None

    def fit(self, X: Tensor, y: Tensor, scores: Tensor | None = None) -> "LDA":
        """Fit LDA projection matrix from historical features.

        Args:
            X: Features (N, D) from all tracks
            y: Track IDs (N,) - which track each feature belongs to
            scores: Detection confidence (N,) - optional weighting

        Returns:
            self for method chaining
        """
        # Convert inputs to correct dtype/device
        X = self._to_tensor(X, self.dtype)
        y = self._to_tensor(y, torch.long)

        if scores is not None:
            scores = self._to_tensor(scores, self.dtype)
            scores = torch.clamp(scores, min=self.weight_min)

        self.classes = torch.unique(y).tolist()
        if len(self.classes) < 2:
            # Cannot compute LDA with <2 classes
            return self

        # Compute class means
        self.class_means = self._compute_class_means(X, y, scores)

        # Within-class covariance (S_w)
        S_w = self._compute_within_class_cov(X, y, scores)

        # Between-class covariance (S_b)
        S_b = self._compute_between_class_cov(X, y, scores)

        # Solve generalized eigenvalue problem: S_b @ v = lambda * S_w @ v
        try:
            eig_vals, eig_vecs = scipy.linalg.eigh(
                S_b.cpu().numpy(), S_w.cpu().numpy()
            )
        except scipy.linalg.LinAlgError:
            # Fallback: add small regularization
            S_w_reg = S_w + 1e-6 * torch.eye(S_w.shape[0], device=self.device, dtype=self.dtype)
            eig_vals, eig_vecs = scipy.linalg.eigh(
                S_b.cpu().numpy(), S_w_reg.cpu().numpy()
            )

        eig_vals = torch.tensor(eig_vals, dtype=self.dtype, device=self.device)
        eig_vecs = torch.tensor(eig_vecs, dtype=self.dtype, device=self.device)

        # Sort by eigenvalue descending, take top (num_classes - 1)
        sorted_idx = torch.argsort(eig_vals, descending=True)
        eig_vecs = eig_vecs[:, sorted_idx]
        self.project_matrix = eig_vecs[:, :len(self.classes) - 1]

        return self

    def transform(self, X: Tensor) -> Tensor:
        """Project features into discriminative subspace.

        Args:
            X: Features (N, D) or (D,)

        Returns:
            Projected features (N, K) where K = num_classes - 1
        """
        if self.project_matrix is None:
            return X  # Return unchanged if not fitted

        X = self._to_tensor(X, self.dtype)
        return X @ self.project_matrix

    def fit_transform(self, X: Tensor, y: Tensor, scores: Tensor | None = None) -> Tensor:
        """Fit and transform in one call."""
        self.fit(X, y, scores)
        return self.transform(X)

    def _to_tensor(self, x: Tensor | list, dtype: torch.dtype) -> Tensor:
        """Convert input to tensor with correct dtype/device."""
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype, device=self.device)
        return torch.tensor(x, dtype=dtype, device=self.device)

    def _compute_class_means(
        self, X: Tensor, y: Tensor, scores: Tensor | None
    ) -> Tensor:
        """Compute mean feature for each class, optionally weighted by scores."""
        class_means = []
        for c in self.classes:
            mask = y == c
            class_X = X[mask]

            if self.use_weighted_class_mean and scores is not None:
                class_scores = scores[mask]
                weights = class_scores ** self.alpha
                mean = (weights[:, None] * class_X).sum(dim=0) / weights.sum()
            else:
                mean = class_X.mean(dim=0)

            class_means.append(mean)

        return torch.stack(class_means, dim=0)

    def _compute_within_class_cov(
        self, X: Tensor, y: Tensor, scores: Tensor | None
    ) -> Tensor:
        """Compute within-class scatter matrix S_w."""
        N, D = X.shape
        S_w = torch.zeros((D, D), dtype=self.dtype, device=self.device)

        for c in self.classes:
            mask = y == c
            class_X = X[mask]
            n_c = mask.sum().item()

            if n_c < 2:
                continue

            cov = self._compute_cov(class_X)

            if self.use_sample_average:
                S_w += (n_c / N) * cov
            else:
                S_w += (1 / len(self.classes)) * cov

        return S_w

    def _compute_between_class_cov(
        self, X: Tensor, y: Tensor, scores: Tensor | None
    ) -> Tensor:
        """Compute between-class scatter matrix S_b."""
        N, D = X.shape
        S_b = torch.zeros((D, D), dtype=self.dtype, device=self.device)

        # Overall mean
        if self.use_weighted_class_mean and scores is not None:
            weights = scores ** self.alpha
            overall_mean = (weights[:, None] * X).sum(dim=0) / weights.sum()
        else:
            overall_mean = X.mean(dim=0)

        for i, c in enumerate(self.classes):
            mask = y == c

            if self.use_weighted_class_mean and scores is not None:
                n = scores[mask].sum()
            else:
                n = mask.sum().float()

            mean_diff = self.class_means[i] - overall_mean

            if self.use_sample_average:
                S_b += n * torch.outer(mean_diff, mean_diff)
            else:
                S_b += (N / len(self.classes)) * torch.outer(mean_diff, mean_diff)

        return S_b

    def _compute_cov(self, X: Tensor) -> Tensor:
        """Compute covariance matrix with optional shrinkage."""
        if not self.use_shrinkage:
            # Simple covariance
            X_centered = X - X.mean(dim=0)
            return X_centered.T @ X_centered / X.shape[0]

        # Shrinkage covariance (Ledoit-Wolf style)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.cpu().numpy())
        X_scaled = torch.tensor(X_scaled, dtype=self.dtype, device=self.device)

        cov = self._compute_shrunk_cov(X_scaled)

        # Undo scaling
        scale = torch.tensor(scaler.scale_, dtype=self.dtype, device=self.device)
        cov = scale[:, None] * cov * scale[None, :]

        return cov

    def _compute_shrunk_cov(self, X: Tensor) -> Tensor:
        """Compute shrinkage covariance (Ledoit-Wolf estimator)."""
        X = X - X.mean(dim=0)
        shrinkage = self._compute_shrinkage(X)
        cov = X.T @ X / X.shape[0]
        mu = torch.trace(cov) / cov.shape[0]

        shrunk_cov = (1 - shrinkage) * cov
        diag_idx = torch.arange(cov.shape[0], device=self.device)
        shrunk_cov[diag_idx, diag_idx] += shrinkage * mu

        return shrunk_cov

    def _compute_shrinkage(self, X: Tensor) -> float:
        """Compute optimal shrinkage coefficient (Ledoit-Wolf)."""
        N, D = X.shape
        X2 = X ** 2

        emp_cov_trace = X2.sum(dim=0) / N
        mu = emp_cov_trace.sum() / D

        beta_ = (X2.T @ X2).sum()
        delta_ = (X.T @ X).pow(2).sum()

        delta_ /= N ** 2
        beta = (1.0 / (D * N)) * (beta_ / N - delta_)
        delta = delta_ - 2.0 * mu * emp_cov_trace.sum() + D * mu ** 2
        delta /= D

        beta = min(beta.item(), delta.item())
        shrinkage = 0.0 if beta == 0 else beta / delta.item()

        return shrinkage
