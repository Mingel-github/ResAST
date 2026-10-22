"""Bounded additive correction from 20 call-level acoustic descriptors."""

import numpy as np
import torch
from torch import nn


def hidden_rms(hidden: torch.Tensor) -> torch.Tensor:
    # RMS(h) = sqrt(mean(h^2) + eps); detach the scale from gradients.
    return (
        torch.sqrt(hidden.float().square().mean(dim=1, keepdim=True) + 1.0e-8)
        .detach()
        .to(hidden.dtype)
    )


class AcousticResidualHead(nn.Module):
    def __init__(
        self, ast_mean: np.ndarray, ast_scale: np.ndarray,
        acoustic_train: np.ndarray, dropout: float,
    ):
        super().__init__()
        if ast_mean.shape != (768,) or ast_scale.shape != (768,):
            raise ValueError("Expected 768-dimensional AST training statistics")
        if acoustic_train.ndim != 2 or acoustic_train.shape[1] != 20 or not len(acoustic_train):
            raise ValueError("Expected training-side acoustic features with shape [calls, 20]")
        safe_scale = np.where(ast_scale > 1.0e-12, ast_scale, 1.0).astype(np.float32)
        self.register_buffer("ast_mean", torch.from_numpy(ast_mean.astype(np.float32)))
        self.register_buffer("ast_scale", torch.from_numpy(safe_scale))
        # Keep the common head initialization before the acoustic branch.
        self.ast_linear = nn.Linear(768, 128)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(128, eps=1.0e-3, momentum=0.01)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(128, 3)
        # Fit imputation and scaling statistics only on the current training partition.
        with np.errstate(all="ignore"):
            median = np.nanmedian(acoustic_train, axis=0)
        median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
        imputed = np.where(np.isfinite(acoustic_train), acoustic_train, median[None, :])
        mean = imputed.mean(axis=0).astype(np.float32)
        scale = imputed.std(axis=0).astype(np.float32)
        scale = np.where(scale > 1.0e-8, scale, 1.0).astype(np.float32)
        self.register_buffer("age_median", torch.from_numpy(median))
        self.register_buffer("age_mean", torch.from_numpy(mean))
        self.register_buffer("age_scale", torch.from_numpy(scale))
        self.age_hidden = nn.Linear(20, 60)
        self.age_output = nn.Linear(60, 128)
        # Start from the exact AST-only path and learn corrections from zero.
        nn.init.zeros_(self.age_output.weight)
        nn.init.zeros_(self.age_output.bias)

    def forward(
        self, ast_embeddings: torch.Tensor, acoustic_features: torch.Tensor
    ) -> torch.Tensor:
        if acoustic_features.shape != (len(ast_embeddings), 20):
            raise ValueError("Acoustic features must align with calls")
        ast = (ast_embeddings - self.ast_mean) / self.ast_scale
        hidden = self.relu(self.ast_linear(ast))
        imputed = torch.where(
            torch.isfinite(acoustic_features), acoustic_features, self.age_median
        )
        standardized = (imputed - self.age_mean) / self.age_scale
        # g(a) = W2 GELU(W1 a + b1) + b2, with widths 20 -> 60 -> 128.
        context = nn.functional.gelu(self.age_hidden(standardized))
        # r = 0.25 * RMS(h) * tanh(g(a)); each coordinate is magnitude-bounded.
        residual = 0.25 * hidden_rms(hidden) * torch.tanh(self.age_output(context))
        # h' = h + r stays 128-dimensional; normalization follows fusion.
        hidden = hidden + residual
        hidden = self.batch_norm(hidden)
        hidden = self.dropout(hidden)
        return self.output(hidden)
