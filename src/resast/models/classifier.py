"""Call-level classification from cached AST tokens."""

import numpy as np
import torch
from torch import nn

from .acoustic_residual import AcousticResidualHead
from .multiscale import ASTTail, average_segments

MODEL_FLAGS = {
    "ast": (False, False),
    "multiscale_ast": (True, False),
    "ast_acoustic": (False, True),
    "resast": (True, True),
}
TRAINABLE_PARAMETERS = {
    "ast": 99_075, "multiscale_ast": 99_099,
    "ast_acoustic": 108_143, "resast": 108_167,
}


class ClassificationHead(nn.Module):
    def __init__(self, mean: np.ndarray, scale: np.ndarray, dropout: float):
        super().__init__()
        if mean.shape != (768,) or scale.shape != (768,):
            raise ValueError("Expected 768-dimensional AST training statistics")
        safe_scale = np.where(scale > 1.0e-12, scale, 1.0).astype(np.float32)
        self.register_buffer("feature_mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("feature_scale", torch.from_numpy(safe_scale))
        self.ast_linear = nn.Linear(768, 128)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(128, eps=1.0e-3, momentum=0.01)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(128, 3)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        normalized = (embeddings - self.feature_mean) / self.feature_scale
        hidden = self.relu(self.ast_linear(normalized))
        hidden = self.dropout(self.batch_norm(hidden))
        return self.output(hidden)


class ResASTClassifier(nn.Module):
    def __init__(self, model_name: str, head: nn.Module, tail: ASTTail):
        super().__init__()
        if model_name not in MODEL_FLAGS:
            raise ValueError(f"Unknown model: {model_name}")
        self.model_name = model_name
        self.head = head
        self.tail = tail

    def forward(
        self, tokens: torch.Tensor, segment_to_call: torch.Tensor,
        call_count: int, acoustic_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        calls = average_segments(self.tail(tokens), segment_to_call, call_count)
        if not MODEL_FLAGS[self.model_name][1]:
            return self.head(calls)
        if acoustic_features is None or acoustic_features.shape != (call_count, 20):
            raise ValueError("Expected call-aligned 20-dimensional acoustic input")
        return self.head(calls, acoustic_features)


def build_classifier(
    model_name: str, backbone: nn.Module, training_embeddings: np.ndarray,
    *, dropout: float, acoustic_train: np.ndarray | None = None,
) -> ResASTClassifier:
    """Construct the head first. The caller controls initialization/training seeds."""
    if model_name not in MODEL_FLAGS:
        raise ValueError(f"Unknown model: {model_name}")
    if training_embeddings.ndim != 2 or training_embeddings.shape[1] != 768 or not len(training_embeddings):
        raise ValueError("Expected original-AST training embeddings with shape [calls, 768]")
    multiscale, acoustic = MODEL_FLAGS[model_name]
    mean = training_embeddings.mean(axis=0)
    scale = training_embeddings.std(axis=0)
    if acoustic:
        if acoustic_train is None or len(acoustic_train) != len(training_embeddings):
            raise ValueError("Training-side acoustic rows must match embedding rows")
        head = AcousticResidualHead(mean, scale, acoustic_train, dropout)
    else:
        head = ClassificationHead(mean, scale, dropout)
    tail = ASTTail(
        backbone.encoder.layer[10], backbone.encoder.layer[11], backbone.layernorm,
        multiscale=multiscale,
    )
    model = ResASTClassifier(model_name, head, tail)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count != TRAINABLE_PARAMETERS[model_name]:
        raise RuntimeError(f"Unexpected trainable parameter count: {count}")
    return model
