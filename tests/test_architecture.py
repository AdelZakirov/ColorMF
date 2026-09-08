from pathlib import Path

import torch
import yaml

from src.model import PixelMeanFlowB, RMSNorm, RoPEAttention, SwiGLUMlp


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
