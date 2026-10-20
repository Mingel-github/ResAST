import os
from types import SimpleNamespace

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch
from transformers import ASTConfig
from transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer import ASTLayer

torch.set_num_threads(2)


@pytest.fixture(scope="session")
def backbone():
    torch.manual_seed(1020)
    config = ASTConfig(
        hidden_size=768, num_attention_heads=12, intermediate_size=3072,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
        max_length=128, num_mel_bins=128,
    )
    config._attn_implementation = "sdpa"
    blocks = [ASTLayer(config).eval(), ASTLayer(config).eval()]
    for block in blocks:
        for parameter in block.parameters():
            parameter.requires_grad = False
    return SimpleNamespace(
        encoder=SimpleNamespace(layer=[None] * 10 + blocks),
        layernorm=torch.nn.LayerNorm(768, eps=config.layer_norm_eps).eval(),
    )
