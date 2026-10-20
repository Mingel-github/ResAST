import numpy as np
import pytest
import torch

from resast.models.acoustic_residual import AcousticResidualHead, hidden_rms
from resast.models.classifier import ClassificationHead, build_classifier, TRAINABLE_PARAMETERS
from resast.models.multiscale import average_segments, pool_patch_kv, multiscale_block


def test_pool_preserves_special_tokens_and_local_neighborhoods():
    values = torch.arange(146, dtype=torch.float32)[None, None, :, None].expand(2, 12, -1, 64)
    pooled = pool_patch_kv(values)
    assert pooled.shape == (2, 12, 38, 64)
    torch.testing.assert_close(pooled[:, :, :2], values[:, :, :2], rtol=0, atol=0)
    # First frequency/time neighborhood consists of patch indices 0,1,12,13.
    assert pooled[0, 0, 2, 0].item() == (2 + 3 + 14 + 15) / 4


def test_zero_gate_matches_native_attention(backbone):
    tokens = torch.randn(2, 146, 768)
    block = backbone.encoder.layer[10]
    native = block(tokens, None, False)[0]
    plain = multiscale_block(block, tokens, None)
    zero = multiscale_block(block, tokens, torch.zeros(12))
    torch.testing.assert_close(plain, native, rtol=0, atol=0)
    torch.testing.assert_close(zero, plain, rtol=0, atol=0)


@pytest.mark.parametrize("name", list(TRAINABLE_PARAMETERS))
def test_model_shape_freeze_and_gradient(backbone, name):
    rng = np.random.default_rng(12)
    embeddings = rng.normal(size=(6, 768)).astype(np.float32)
    acoustics = rng.normal(size=(6, 20)).astype(np.float32)
    torch.manual_seed(17)
    model = build_classifier(
        name, backbone, embeddings, dropout=0.25, acoustic_train=acoustics
    )
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == TRAINABLE_PARAMETERS[name]
    model.train()
    assert model.head.training
    assert not model.tail.training
    tokens = torch.randn(3, 146, 768)
    logits = model(tokens, torch.tensor([0, 1, 1]), 2, torch.from_numpy(acoustics[:2]))
    assert logits.shape == (2, 3)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2])).backward()
    assert model.head.ast_linear.weight.grad is not None
    assert all(p.grad is None for p in model.tail.block11.parameters())
    assert all(p.grad is None for p in model.tail.block12.parameters())
    assert all(p.grad is None for p in model.tail.final_norm.parameters())
    if model.tail.raw_gamma is not None:
        assert model.tail.raw_gamma.grad is not None
        assert torch.isfinite(model.tail.raw_gamma.grad).all()


def test_residual_starts_at_zero_and_preserves_common_head():
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(7, 768)).astype(np.float32)
    acoustics = rng.normal(size=(7, 20)).astype(np.float32)
    torch.manual_seed(43)
    plain = ClassificationHead(embeddings.mean(0), embeddings.std(0), 0.25).eval()
    torch.manual_seed(43)
    residual = AcousticResidualHead(
        embeddings.mean(0), embeddings.std(0), acoustics, 0.25
    ).eval()
    for name, parameter in plain.named_parameters():
        torch.testing.assert_close(parameter, dict(residual.named_parameters())[name], rtol=0, atol=0)
    inputs = torch.from_numpy(embeddings)
    torch.testing.assert_close(
        plain(inputs), residual(inputs, torch.from_numpy(acoustics)), rtol=0, atol=0
    )


def test_rms_detached_and_residual_coordinate_bound():
    hidden = torch.randn(5, 128, requires_grad=True)
    scale = hidden_rms(hidden)
    assert not scale.requires_grad
    residual = 0.25 * scale * torch.tanh(torch.randn_like(hidden) * 10)
    assert bool(torch.all(residual.abs() <= 0.25 * scale))


def test_training_only_acoustic_statistics_and_missing_values():
    acoustic = np.ones((6, 20), np.float32)
    acoustic[0, 0] = np.nan
    acoustic[:, 1] = np.nan
    with pytest.warns(RuntimeWarning, match="All-NaN"):
        head = AcousticResidualHead(np.zeros(768), np.ones(768), acoustic, 0.0).eval()
    assert head.age_median[0].item() == 1
    assert head.age_median[1].item() == 0
    assert head.age_scale.eq(1).all()
    before = head.age_mean.clone()
    outputs = head(torch.randn(3, 768), torch.full((3, 20), float("nan")))
    assert torch.isfinite(outputs).all()
    torch.testing.assert_close(before, head.age_mean, rtol=0, atol=0)


def test_segment_mean_preserves_call_order():
    features = torch.tensor([1.0, 3.0, 9.0])[:, None].repeat(1, 768)
    result = average_segments(features, torch.tensor([1, 1, 0]), 2)
    torch.testing.assert_close(result[:, 0], torch.tensor([9.0, 2.0]))
    with pytest.raises(ValueError, match="Every call"):
        average_segments(features, torch.tensor([0, 0, 0]), 2)


def test_reject_incorrect_geometry():
    with pytest.raises(ValueError):
        pool_patch_kv(torch.zeros(1, 12, 144, 64))
