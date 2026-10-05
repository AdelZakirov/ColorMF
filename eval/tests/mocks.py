"""Small deterministic neural stubs, solely for technical offline validation."""

import torch


class MockDistance(torch.nn.Module):
    """Mean square distance in normalized RGB; explicitly NOT real LPIPS."""
    def forward(self, first, second):
        self.last_first = first.detach().cpu().clone()
        return (first - second).square().mean(dim=(1, 2, 3), keepdim=True)


class MockFeatures(torch.nn.Module):
    """Small deterministic RGB features; explicitly NOT Inception features."""
    def forward(self, inputs):
        self.last_inputs = inputs.detach().cpu().clone()
        return [inputs.mean(dim=(2, 3), keepdim=True)]
