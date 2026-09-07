import torch
from torch import nn

from src.ema import ExponentialMovingAverage
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


def test_lightning_module_updates_ema_after_optimizer_step():
    module = PMFColorizerModule(
        model={
            "resolution": 16,
            "patch_size": 4,
            "hidden_size": 32,
            "depth": 1,
            "heads": 4,
        },
        ema_decay=0.5,
    )
    assert module.ema is not None
    module.ema.initialize(module.model)
    initial_weight = module.model.clean_head.weight.detach().clone()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.0)

    def closure():
        with torch.no_grad():
            module.model.clean_head.weight.fill_(2.0)
        return torch.zeros((), requires_grad=True)

    module.optimizer_step(0, 0, optimizer, closure)
    assert module.ema.num_updates == 1
    torch.testing.assert_close(
        module.ema._shadow["clean_head.weight"],
        torch.full_like(initial_weight, 2.0),
    )