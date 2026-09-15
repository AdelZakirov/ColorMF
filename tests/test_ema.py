import torch
from torch import nn

from src.ema import EMAManager, ExponentialMovingAverage
from src.lightning_module import PMFColorizerModule


def test_ema_updates_after_configured_warmup_and_interval():
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    ema = ExponentialMovingAverage(
        decay=0.5,
        update_after_step=1,
        update_every=2,
    )
    ema.initialize(model)

    with torch.no_grad():
        model.weight.fill_(2.0)
    ema.update(model)
    assert ema.num_updates == 1

    with torch.no_grad():
        model.weight.fill_(4.0)
    ema.update(model)
    assert ema.num_updates == 2
    assert ema.ready
    torch.testing.assert_close(
        ema._shadow["weight"], torch.full_like(model.weight, 4.0)
    )

    with torch.no_grad():
        model.weight.fill_(6.0)
    ema.update(model)
    assert ema.num_updates == 3

    with torch.no_grad():
        model.weight.fill_(8.0)
    ema.update(model)
    assert ema.num_updates == 4

    original = model.weight.detach().clone()
    ema.store(model)
    ema.copy_to(model)
    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 6.0))
    ema.restore(model)
    torch.testing.assert_close(model.weight, original)


def test_ema_state_roundtrip_preserves_shadow_and_step_count():
    model = nn.Linear(2, 1)
    ema = ExponentialMovingAverage(decay=0.9)
    ema.initialize(model)
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)

    restored = ExponentialMovingAverage(decay=0.9)
    restored.load_state_dict(ema.state_dict())

    assert restored.num_updates == ema.num_updates
    restored.copy_to(model)
    torch.testing.assert_close(model.weight, ema._shadow["weight"])


def test_edm_ema_uses_actual_images_and_roundtrips_all_variants():
    module = PMFColorizerModule(
        model={
            "resolution": 16,
            "patch_size": 4,
            "hidden_size": 32,
            "depth": 2,
            "heads": 4,
            "aux_head_depth": 1,
            "pca_channels": 8,
        },
        ema_type="edm",
        ema_half_lives_kimg=(500, 1000, 2000),
    )
    assert module.ema is not None
    module.ema.initialize(module.model)
    with torch.no_grad():
        module.model.u_final_layer.linear._flax_linear.weight.fill_(2.0)
    module.ema.update(module.model, global_images=256)
    assert module.ema.num_updates == 1
    assert module.ema.images_seen == 256
    assert module.ema.variants == ("500", "1000", "2000")
    restored = EMAManager(ema_type="edm", half_lives_kimg=(500, 1000, 2000))
    restored.load_state_dict(module.ema.state_dict())
    assert restored.images_seen == 256
    for variant in restored.variants:
        restored.copy_to(module.model, variant)


def test_faithful_module_configures_muon():
    module = PMFColorizerModule(
        model={"resolution": 16, "patch_size": 4, "hidden_size": 32,
               "depth": 2, "heads": 4, "aux_head_depth": 1,
               "pca_channels": 8},
        optimizer="muon",
    )
    from src.optimizer import Muon
    assert isinstance(module.configure_optimizers()["optimizer"], Muon)
