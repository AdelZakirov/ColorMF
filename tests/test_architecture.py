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
