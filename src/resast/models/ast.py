"""Loading, positional geometry and the frozen ten-block AST prefix."""

import copy
from pathlib import Path

import torch
from torch import nn

from resast.data.audio import CHECKPOINT, REVISION


def adapt_geometry(
    model: nn.Module,
    max_length: int = 128,
    frequency_stride: int = 10,
    time_stride: int = 10,
) -> dict:
    config = model.config
    embeddings = model.embeddings
    patch_size = int(config.patch_size)
    old_frequency = (config.num_mel_bins - patch_size) // config.frequency_stride + 1
    old_time = (config.max_length - patch_size) // config.time_stride + 1
    new_frequency = (config.num_mel_bins - patch_size) // frequency_stride + 1
    new_time = (max_length - patch_size) // time_stride + 1
    projection_weight = embeddings.patch_embeddings.projection.weight.detach().clone()
    projection_bias = embeddings.patch_embeddings.projection.bias.detach().clone()
    special_positions = embeddings.position_embeddings[:, :2]
    patch_positions = embeddings.position_embeddings[:, 2:]
    hidden_size = patch_positions.shape[-1]
    patch_positions = patch_positions.reshape(1, old_frequency, old_time, hidden_size)
    patch_positions = patch_positions.permute(0, 3, 1, 2)
    patch_positions = nn.functional.interpolate(
        patch_positions,
        size=(new_frequency, new_time),
        mode="bilinear",
        align_corners=False,
    )
    patch_positions = patch_positions.permute(0, 2, 3, 1).reshape(
        1, new_frequency * new_time, hidden_size
    )
    embeddings.position_embeddings = nn.Parameter(
        torch.cat((special_positions, patch_positions), dim=1)
    )
    embeddings.patch_embeddings.projection.stride = (frequency_stride, time_stride)
    config.max_length = max_length
    config.frequency_stride = frequency_stride
    config.time_stride = time_stride
    embeddings.config = config
    reused = torch.equal(projection_weight, embeddings.patch_embeddings.projection.weight)
    reused = reused and torch.equal(projection_bias, embeddings.patch_embeddings.projection.bias)
    if not reused:
        raise RuntimeError("Patch projection changed during position interpolation")
    for parameter in model.parameters():
        parameter.requires_grad = False
    return {
        "source_grid": [old_frequency, old_time],
        "target_grid": [new_frequency, new_time],
        "patch_tokens": new_frequency * new_time,
        "total_tokens": new_frequency * new_time + 2,
    }


def load_backbone(
    source: str | Path = CHECKPOINT,
    *,
    cache_dir: str | Path | None = None,
    local_files_only: bool = True,
) -> nn.Module:
    from transformers import ASTModel

    model = ASTModel.from_pretrained(
        str(source),
        revision=REVISION,
        cache_dir=cache_dir,
        use_safetensors=True,
        local_files_only=local_files_only,
        attn_implementation="sdpa",
    ).cpu().eval()
    if len(model.encoder.layer) != 12 or model.config.hidden_size != 768:
        raise ValueError("Expected the 12-block, 768-dimensional AST backbone")
    adapt_geometry(model)
    return model


class FrozenASTPrefix(nn.Module):
    """Produce [segments, 146, 768] block-ten outputs in float32."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        if len(backbone.encoder.layer) != 12:
            raise ValueError("Expected a 12-block backbone")
        self.embeddings = copy.deepcopy(backbone.embeddings)
        self.blocks = nn.ModuleList(copy.deepcopy(list(backbone.encoder.layer[:10])))
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or tuple(features.shape[1:]) != (128, 128):
            raise ValueError("Expected [segments, 128, 128] filterbank inputs")
        # Prefix caches were generated in float32; tail training uses its own AMP context.
        with torch.autocast(device_type=features.device.type, enabled=False):
            hidden = self.embeddings(features.float())
            for block in self.blocks:
                hidden = block(hidden, None, False)[0]
        return hidden
