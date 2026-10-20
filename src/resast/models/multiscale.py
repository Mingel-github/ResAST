"""Local K/V aggregation with unchanged query resolution."""

import copy

import torch
from torch import nn


def pool_patch_kv(values: torch.Tensor) -> torch.Tensor:
    """Pool 2x2 patch neighborhoods: [B, 12, 146, 64] -> [B, 12, 38, 64]."""
    if values.ndim != 4 or tuple(values.shape[1:]) != (12, 146, 64):
        raise ValueError("Expected 12 heads, 146 tokens and 64 features per head")
    patches = values[:, :, 2:]
    batch, heads, _, features = patches.shape
    coarse = patches.reshape(batch, heads, 6, 2, 6, 2, features)
    coarse = coarse.mean(dim=(3, 5)).reshape(batch, heads, 36, features)
    return torch.cat((values[:, :, :2], coarse), dim=2)


def multiscale_block(
    block: nn.Module, hidden: torch.Tensor, gamma: torch.Tensor | None
) -> torch.Tensor:
    if hidden.ndim != 3 or tuple(hidden.shape[1:]) != (146, 768):
        raise ValueError("Expected [segments, 146, 768] tokens")
    if gamma is not None and gamma.shape != (12,):
        raise ValueError("Expected one scale coefficient per head")
    if block.training:
        raise ValueError("Frozen AST blocks must remain in evaluation mode")
    normalized = block.layernorm_before(hidden)
    attention = block.attention.attention
    if (
        attention.num_attention_heads != 12
        or attention.attention_head_size != 64
        or attention.attention_probs_dropout_prob != 0.0
    ):
        raise ValueError("Unexpected AST attention geometry or dropout")
    mixed_query = attention.query(normalized)
    key = attention.transpose_for_scores(attention.key(normalized))
    value = attention.transpose_for_scores(attention.value(normalized))
    query = attention.transpose_for_scores(mixed_query)
    fine = nn.functional.scaled_dot_product_attention(
        query, key, value, None, 0.0, is_causal=False, scale=None
    )
    context = fine
    if gamma is not None:
        coarse = nn.functional.scaled_dot_product_attention(
            query, pool_patch_kv(key), pool_patch_kv(value),
            None, 0.0, is_causal=False, scale=None,
        )
        alpha = torch.tanh(gamma).to(fine.dtype).reshape(1, 12, 1, 1)
        context = fine + alpha * (coarse - fine)
    context = context.permute(0, 2, 1, 3).contiguous().view(hidden.shape[0], 146, 768)
    attended = block.attention.output(context, normalized) + hidden
    return block.output(block.intermediate(block.layernorm_after(attended)), attended)


class ASTTail(nn.Module):
    """Frozen blocks eleven/twelve, optionally with 24 trainable coefficients."""

    def __init__(
        self, block11: nn.Module, block12: nn.Module, final_norm: nn.Module,
        *, multiscale: bool,
    ):
        super().__init__()
        self.block11 = copy.deepcopy(block11)
        self.block12 = copy.deepcopy(block12)
        self.final_norm = copy.deepcopy(final_norm)
        for module in (self.block11, self.block12, self.final_norm):
            for parameter in module.parameters():
                parameter.requires_grad = False
        if multiscale:
            self.raw_gamma = nn.Parameter(torch.zeros(2, 12, dtype=torch.float32))
        else:
            self.register_parameter("raw_gamma", None)
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)

    def sequence(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = tokens
        for index, block in enumerate((self.block11, self.block12)):
            gamma = None if self.raw_gamma is None else self.raw_gamma[index]
            hidden = multiscale_block(block, hidden, gamma)
        return self.final_norm(hidden)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        sequence = self.sequence(tokens)
        return (sequence[:, 0] + sequence[:, 1]) / 2.0


def average_segments(
    segments: torch.Tensor, segment_to_call: torch.Tensor, call_count: int
) -> torch.Tensor:
    if segments.ndim != 2 or segments.shape[1] != 768:
        raise ValueError("Expected [segments, 768] features")
    if segment_to_call.shape != (len(segments),) or segment_to_call.dtype != torch.long:
        raise ValueError("Expected one int64 call index per segment")
    if call_count < 1 or not len(segments):
        raise ValueError("At least one call and segment are required")
    if bool(torch.any(segment_to_call < 0)) or bool(torch.any(segment_to_call >= call_count)):
        raise ValueError("Segment mapping is outside the current batch")
    calls = torch.zeros(call_count, 768, dtype=segments.dtype, device=segments.device)
    calls.index_add_(0, segment_to_call, segments)
    counts = torch.bincount(segment_to_call, minlength=call_count).to(segments.dtype)
    if not bool(torch.all(counts > 0)):
        raise ValueError("Every call must have at least one segment")
    return calls / counts[:, None]
