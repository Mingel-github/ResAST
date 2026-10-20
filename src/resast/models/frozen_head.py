"""Classification head for frozen CNN14, AVES and SSAST call embeddings."""

import torch
from torch import nn

from resast.training.common import standardizer

DIMENSIONS = {"cnn14": 2048, "aves": 768, "ssast": 768}


class FrozenEmbeddingHead(nn.Module):
    def __init__(self, training_embeddings, dropout):
        super().__init__()
        mean, scale = standardizer(training_embeddings)
        self.register_buffer("embedding_mean", torch.from_numpy(mean))
        self.register_buffer("embedding_scale", torch.from_numpy(scale))
        self.embedding_linear = nn.Linear(training_embeddings.shape[1], 128)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(128, eps=1e-3, momentum=0.01)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(128, 3)

    def forward(self, embeddings):
        hidden = self.relu(self.embedding_linear((embeddings - self.embedding_mean) / self.embedding_scale))
        return self.output(self.dropout(self.batch_norm(hidden)))
