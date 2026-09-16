from pathlib import Path

import torch
import yaml

from src.model import (
    BottleneckPatchEmbedder,
    PixelMeanFlowB,
    RMSNorm,
    RoPEAttention,
    SwiGLUMlp,
    TorchLinear,
)


def tiny_model():
    return PixelMeanFlowB(resolution=16, patch_size=4, hidden_size=32,
                          depth=4, num_heads=4, aux_head_depth=2,
                          pca_channels=8)


def test_pmf_components_branches_gates_and_zero_final_layers():
    model = tiny_model()
    assert len(model.shared_blocks) == len(model.u_blocks) == len(model.v_blocks) == 2
    assert isinstance(model.shared_blocks[0].norm1, RMSNorm)
    assert isinstance(model.shared_blocks[0].attn, RoPEAttention)
    assert isinstance(model.shared_blocks[0].mlp, SwiGLUMlp)
    for block in [*model.shared_blocks, *model.u_blocks, *model.v_blocks]:
        assert torch.count_nonzero(block.attn_scale) == 0
        assert torch.count_nonzero(block.mlp_scale) == 0
    assert torch.count_nonzero(model.u_final_layer.linear._flax_linear.weight) == 0
    assert torch.count_nonzero(model.v_final_layer.linear._flax_linear.weight) == 0
    report = model.parameter_report()
    assert report["total_training_parameters"] == (
        report["shared_parameters"] + report["u_head_parameters"] + report["v_head_parameters"])


def test_rope_preserves_prefix_tokens():
    attention = tiny_model().shared_blocks[0].attn
    x = torch.randn(1, 20, 4, 8)
    from src.model import apply_rotary_pos_emb, precompute_rope_freqs
    rope = precompute_rope_freqs(8, 16)
    rotated = apply_rotary_pos_emb(x, rope)
    torch.testing.assert_close(rotated[:, :4], x[:, :4])
    assert not torch.equal(rotated[:, 4:], x[:, 4:])


def test_no_grad_attention_uses_explicit_softmax():
    attention = tiny_model().shared_blocks[0].attn.eval()
    from src.model import apply_rotary_pos_emb, precompute_rope_freqs
    x = torch.randn(2, 20, 32)
    rope = precompute_rope_freqs(8, 16)
    shape = (2, 20, attention.num_heads, attention.head_dim)
    with torch.no_grad():
        q = apply_rotary_pos_emb(
            attention.q_norm(attention.q_proj(x).reshape(shape)), rope)
        k = apply_rotary_pos_emb(
            attention.k_norm(attention.k_proj(x).reshape(shape)), rope)
        v = attention.v_proj(x).reshape(shape)
        weights = torch.einsum("bqhd,bkhd->bhqk", q / attention.head_dim ** 0.5, k)
        weights = torch.softmax(weights, dim=-1)
        expected = attention.out_proj(torch.einsum(
            "bhqk,bkhd->bqhd", weights, v).reshape(2, 20, 32))
        actual = attention(x, rope)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_auxiliary_direction_matches_full_forward_v_branch():
    model = tiny_model().eval()
    with torch.no_grad():
        for block in [*model.shared_blocks, *model.v_blocks]:
            block.attn_scale.normal_(std=0.02)
            block.mlp_scale.normal_(std=0.02)
        model.v_final_layer.linear._flax_linear.weight.normal_(std=0.01)
        model.v_final_layer.linear._flax_linear.bias.normal_(std=0.01)
    z = torch.randn(2, 2, 16, 16)
    L = torch.randn(2, 1, 16, 16)
    t = torch.tensor([0.2, 0.7])
    with torch.no_grad():
        expected = model(z, L, t, t)[1]
        actual = model.auxiliary_direction(z, L, t)
    torch.testing.assert_close(actual, expected)


def test_faithful_configs_have_256_spatial_tokens_and_recipe():
    root = Path(__file__).resolve().parents[1]
    for resolution, patch in ((64, 4), (128, 8), (256, 16)):
        config = yaml.safe_load((root / "configs" / f"pmf_b_{resolution}_colorization.yaml").read_text())
        model = config["model"]
        assert (resolution // patch) ** 2 == 256
        assert model["resolution"] == resolution and model["patch_size"] == patch
        assert model["in_channels"] == 3 and model["out_channels"] == 2
        assert model["depth"] == 16 and model["aux_head_depth"] == 8
        assert config["training"]["optimizer"] == "muon"
        assert config["training"]["ema"]["half_lives_kimg"] == [500, 1000, 2000]


def test_inference_does_not_execute_v_branch():
    model = tiny_model().eval()
    calls = []
    handle = model.v_blocks[0].register_forward_hook(lambda *args: calls.append(1))
    model.sample(torch.zeros(1, 1, 16, 16), seed=1, image_ids=["x"])
    handle.remove()
    assert calls == []
    assert model.last_sample_nfe == 1


def test_concat_mode_preserves_legacy_state_dict_and_ignores_reinject():
    kwargs = dict(resolution=16, patch_size=4, hidden_size=32, depth=4,
                  num_heads=4, aux_head_depth=2, pca_channels=8)
    torch.manual_seed(123)
    legacy = PixelMeanFlowB(**kwargs)
    torch.manual_seed(123)
    explicit = PixelMeanFlowB(**kwargs, conditioning={"mode": "concat", "reinject": False})
    assert isinstance(explicit.x_embedder, BottleneckPatchEmbedder)
    assert not any("condition" in key or "state_embedder" in key
                   for key in explicit.state_dict())
    assert list(legacy.state_dict()) == list(explicit.state_dict())
    for key, value in legacy.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key])
    z = torch.randn(2, 2, 16, 16)
    L = torch.randn(2, 1, 16, 16)
    r, t = torch.tensor([0.1, 0.2]), torch.tensor([0.7, 0.8])
    torch.testing.assert_close(legacy(z, L, r, t)[0], explicit(z, L, r, t)[0])


def test_separate_conditioning_has_aligned_embedders_and_zero_gates():
    model = PixelMeanFlowB(
        resolution=16, patch_size=4, hidden_size=32, depth=4, num_heads=4,
        aux_head_depth=2, pca_channels=8,
        conditioning={"mode": "separate", "reinject": True},
    )
    assert not hasattr(model, "x_embedder")
    assert isinstance(model.state_embedder, BottleneckPatchEmbedder)
    assert isinstance(model.condition_embedder, BottleneckPatchEmbedder)
    assert model.state_embedder is not model.condition_embedder
    assert model.state_embedder.proj1.in_channels == 2
    assert model.condition_embedder.proj1.in_channels == 1
    assert model.state_embedder.patch_size == model.condition_embedder.patch_size
    assert model.state_embedder.grid_size == model.condition_embedder.grid_size
    assert model.num_spatial_tokens == model.state_embedder.num_patches == 16
    assert isinstance(model.condition_norm, RMSNorm)
    assert isinstance(model.condition_proj, TorchLinear)
    assert model.condition_proj._flax_linear.bias is None
    for gates, blocks in (
        (model.shared_condition_gates, model.shared_blocks),
        (model.u_condition_gates, model.u_blocks),
        (model.v_condition_gates, model.v_blocks),
    ):
        assert len(gates) == len(blocks)
        assert all(gate.shape == (model.hidden_size,) for gate in gates)
        assert all(torch.count_nonzero(gate) == 0 for gate in gates)


def test_separate_initial_fusion_and_condition_sequence_are_spatially_aligned():
    model = PixelMeanFlowB(
        resolution=16, patch_size=4, hidden_size=32, depth=4, num_heads=4,
        aux_head_depth=2, pca_channels=8,
        conditioning={"mode": "separate", "reinject": True},
    )
    z = torch.randn(2, 2, 16, 16)
    L = torch.randn(2, 1, 16, 16)
    h = torch.tensor([0.2, 0.7])
    sequence, condition = model._sequence(z, L, h)
    state = model.state_embedder(z)
    projected = model.condition_proj(model.condition_norm(model.condition_embedder(L)))
    expected = torch.cat([model.time_tokens + model.h_embedder(h)[:, None],
                          state + projected], dim=1) + model.pos_embed
    torch.testing.assert_close(sequence, expected)
    assert condition.shape == sequence.shape
    assert torch.count_nonzero(condition[:, :model.prefix_tokens]) == 0
    torch.testing.assert_close(condition[:, model.prefix_tokens:], projected)


def test_separate_reinjection_is_reused_by_auxiliary_direction_and_keeps_l_gradients():
    model = PixelMeanFlowB(
        resolution=16, patch_size=4, hidden_size=32, depth=4, num_heads=4,
        aux_head_depth=2, pca_channels=8,
        conditioning={"mode": "separate", "reinject": True},
    ).eval()
    with torch.no_grad():
        for block in [*model.shared_blocks, *model.u_blocks, *model.v_blocks]:
            block.attn_scale.normal_(std=0.02)
            block.mlp_scale.normal_(std=0.02)
        for gates in [model.shared_condition_gates, model.u_condition_gates,
                      model.v_condition_gates]:
            for gate in gates:
                gate.normal_(std=0.02)
        model.v_final_layer.linear._flax_linear.weight.normal_(std=0.01)
        model.v_final_layer.linear._flax_linear.bias.normal_(std=0.01)
        model.u_final_layer.linear._flax_linear.weight.normal_(std=0.01)
    z = torch.randn(2, 2, 16, 16)
    L = torch.randn(2, 1, 16, 16, requires_grad=True)
    t = torch.tensor([0.2, 0.7])
    with torch.no_grad():
        expected = model(z, L, t, t)[1]
        actual = model.auxiliary_direction(z, L, t)
    torch.testing.assert_close(actual, expected)
    u, _ = model(z, L, t, t, return_velocity=False)
    u.square().mean().backward()
    assert torch.count_nonzero(L.grad) > 0
    assert torch.count_nonzero(model.condition_proj._flax_linear.weight.grad) > 0


def test_separate_without_reinjection_has_no_gate_parameters_and_reports_all_parameters():
    model = PixelMeanFlowB(
        resolution=16, patch_size=4, hidden_size=32, depth=4, num_heads=4,
        aux_head_depth=2, pca_channels=8,
        conditioning={"mode": "separate", "reinject": False},
    )
    assert model.shared_condition_gates is None
    assert model.u_condition_gates is None
    assert model.v_condition_gates is None
    assert not any("condition_gates" in key for key in model.state_dict())
    report = model.parameter_report()
    assert report["total_training_parameters"] == (
        report["shared_parameters"] + report["u_head_parameters"]
        + report["v_head_parameters"])
