"""Differentiable loss functions used for colorization-error comparison.

The evaluation input is RGB, but the two direct color losses operate on the
physical CIELAB ``a*`` and ``b*`` channels.  The feature losses operate on
RGB after their model-specific input preprocessing.  All feature networks
are frozen; gradients are retained with respect to the predicted image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from kornia.color import rgb_to_lab


LOSS_NAMES = (
    "huber_ab",
    "gradient_ab",
    "lpips_vgg",
    "convnext_perceptual",
    "dinov3_perceptual",
)


def physical_lab(rgb: Tensor) -> Tensor:
    """Convert RGB in ``[0, 1]`` to physical CIELAB ``[L*, a*, b*]``."""

    return rgb_to_lab(rgb)


def huber_ab_loss(pred_rgb: Tensor, target_rgb: Tensor, beta: float = 5.0) -> Tensor:
    """Mean Smooth-L1 distance between physical Lab chroma channels."""

    pred_ab = physical_lab(pred_rgb)[:, 1:]
    target_ab = physical_lab(target_rgb)[:, 1:]
    return F.smooth_l1_loss(pred_ab, target_ab, beta=beta, reduction="mean")


def _ab_gradients(ab: Tensor) -> tuple[Tensor, Tensor]:
    """Forward horizontal and vertical finite differences of Lab chroma."""

    return ab[..., :, 1:] - ab[..., :, :-1], ab[..., 1:, :] - ab[..., :-1, :]


def gradient_ab_loss(pred_rgb: Tensor, target_rgb: Tensor) -> Tensor:
    """Mean L1 difference of horizontal/vertical ``a*, b*`` gradients."""

    pred_dx, pred_dy = _ab_gradients(physical_lab(pred_rgb)[:, 1:])
    target_dx, target_dy = _ab_gradients(physical_lab(target_rgb)[:, 1:])
    horizontal = (pred_dx - target_dx).abs().mean()
    vertical = (pred_dy - target_dy).abs().mean()
    return 0.5 * (horizontal + vertical)


def _freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def spatial_distance(predicted, target):
    """Equal-weight mean MSE across corresponding spatial feature stages."""
    if not predicted or len(predicted) != len(target):
        raise ValueError("Corresponding nonempty feature stages are required")
    return torch.stack([(p.float() - t.float()).square().flatten(1).mean(1)
                        for p, t in zip(predicted, target)]).mean(0)


def _resize_rgb(rgb: Tensor, size: int = 224) -> Tensor:
    if rgb.shape[-2:] == (size, size):
        return rgb
    return F.interpolate(rgb, size=(size, size), mode="bicubic", align_corners=False, antialias=True)


@dataclass(frozen=True)
class LossConfig:
    """Model and numerical settings recorded in the experiment manifest."""

    huber_beta: float = 5.0
    feature_size: int = 224
    convnext_input_range: str = "processor_mean_std"
    dino_layers: tuple[int, ...] = (3, 6, 9, 12)
    dino_model: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    convnext_model: str = "facebook/convnextv2-base-22k-224"


class LossSuite(nn.Module):
    """Five losses with a common differentiable RGB prediction interface."""

    def __init__(
        self,
        *,
        device: torch.device,
        config: LossConfig = LossConfig(),
        dino_model: Optional[str] = None,
        convnext_model: Optional[str] = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        dino_model = dino_model or config.dino_model
        convnext_model = convnext_model or config.convnext_model

        # These imports are deliberately lazy.  The direct Lab losses and unit
        # tests remain usable in the lightweight training environment, while
        # the evaluation script can run with transformers 5.17 for DINOv3.
        import lpips
        from transformers import AutoImageProcessor, ConvNextV2Model, DINOv3ViTModel

        self.lpips = _freeze(lpips.LPIPS(net="vgg"))
        self.convnext = _freeze(ConvNextV2Model.from_pretrained(
            convnext_model, local_files_only=local_files_only))
        # Published preprocessor_config.json for this exact checkpoint. Keeping
        # these constants permits offline evaluation when only weights are cached.
        if convnext_model == "facebook/convnextv2-base-22k-224":
            convnext_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            processor = AutoImageProcessor.from_pretrained(
                convnext_model, local_files_only=local_files_only)
            convnext_stats = (processor.image_mean, processor.image_std)
        for name, values in zip(("mean", "std"), convnext_stats):
            self.register_buffer("convnext_" + name, torch.tensor(values).view(1, 3, 1, 1))

        self.dino_processor = AutoImageProcessor.from_pretrained(
            dino_model, local_files_only=local_files_only)
        self.dino = _freeze(DINOv3ViTModel.from_pretrained(
            dino_model, local_files_only=local_files_only))
        dino_mean = getattr(self.dino_processor, "image_mean", None)
        dino_std = getattr(self.dino_processor, "image_std", None)
        if dino_mean is None or dino_std is None:
            raise RuntimeError("DINOv3 image processor does not expose image_mean/image_std")
        self.register_buffer("dino_mean", torch.tensor(dino_mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("dino_std", torch.tensor(dino_std, dtype=torch.float32).view(1, 3, 1, 1))
        self.to(device)

    def _lpips_distance(self, predicted_rgb: Tensor, target_rgb: Tensor) -> Tensor:
        predicted = _resize_rgb(predicted_rgb, self.config.feature_size) * 2.0 - 1.0
        target = _resize_rgb(target_rgb, self.config.feature_size) * 2.0 - 1.0
        return self.lpips(predicted, target).reshape(predicted.shape[0], -1).mean(1)

    def _convnext_features(self, rgb: Tensor) -> Tensor:
        x = (_resize_rgb(rgb, self.config.feature_size) - self.convnext_mean) / self.convnext_std
        return self.convnext(pixel_values=x, output_hidden_states=True).hidden_states[1:]

    def _dino_features(self, rgb: Tensor) -> tuple[Tensor, ...]:
        x = (_resize_rgb(rgb, self.config.feature_size) - self.dino_mean) / self.dino_std
        output = self.dino(pixel_values=x, output_hidden_states=True)
        start = 1 + self.dino.config.num_register_tokens
        return tuple(F.normalize(output.hidden_states[i][:, start:].float(), dim=-1)
                     for i in self.config.dino_layers)

    def loss_ab(self, name: str, predicted_ab: Tensor, target_ab: Tensor, L: Tensor) -> Tensor:
        """Physical ab leaf coordinates, fixed L; no RGB roundtrip for direct losses."""
        if name == "huber_ab":
            return F.smooth_l1_loss(predicted_ab, target_ab, beta=self.config.huber_beta,
                                   reduction="none").mean((1, 2, 3))
        if name == "gradient_ab":
            px, py = _ab_gradients(predicted_ab)
            tx, ty = _ab_gradients(target_ab)
            return .5 * ((px-tx).abs().mean((1,2,3)) + (py-ty).abs().mean((1,2,3)))
        from kornia.color import lab_to_rgb
        return self.loss(name, lab_to_rgb(torch.cat((L, predicted_ab), 1), clip=False),
                         lab_to_rgb(torch.cat((L, target_ab), 1), clip=False))

    def loss(self, loss_name: str, predicted_rgb: Tensor, target_rgb: Tensor) -> Tensor:
        """Compute one per-example loss vector.

        The evaluation runner calls this one loss at a time to avoid keeping
        all five feature-model activation graphs in GPU memory simultaneously.
        """

        if loss_name == "huber_ab":
            pred_ab = physical_lab(predicted_rgb)[:, 1:]
            target_ab = physical_lab(target_rgb)[:, 1:]
            return F.smooth_l1_loss(
                pred_ab, target_ab, beta=self.config.huber_beta, reduction="none").mean((1, 2, 3))
        if loss_name == "gradient_ab":
            pred_ab = physical_lab(predicted_rgb)[:, 1:]
            target_ab = physical_lab(target_rgb)[:, 1:]
            pred_dx, pred_dy = _ab_gradients(pred_ab)
            target_dx, target_dy = _ab_gradients(target_ab)
            return 0.5 * (
                (pred_dx - target_dx).abs().mean((1, 2, 3))
                + (pred_dy - target_dy).abs().mean((1, 2, 3)))
        if loss_name == "lpips_vgg":
            return self._lpips_distance(predicted_rgb, target_rgb.detach())
        if loss_name == "convnext_perceptual":
            predicted = self._convnext_features(predicted_rgb)
            with torch.no_grad():
                target = self._convnext_features(target_rgb.detach())
            return spatial_distance(predicted, target)
        if loss_name == "dinov3_perceptual":
            predicted = self._dino_features(predicted_rgb)
            with torch.no_grad():
                target = self._dino_features(target_rgb.detach())
            return torch.stack([(1 - (p * t).sum(-1)).mean(1)
                                for p, t in zip(predicted, target)]).mean(0)
        raise KeyError(f"Unknown loss: {loss_name}")

    def forward(self, predicted_rgb: Tensor, target_rgb: Tensor) -> Dict[str, Tensor]:
        """Return one value per batch item and loss.

        Returning vectors rather than already-reduced scalars lets the runner
        obtain an independent image gradient for every item in a batch by
        differentiating ``values[name].sum()``.
        """

        return {loss_name: self.loss(loss_name, predicted_rgb, target_rgb) for loss_name in LOSS_NAMES}

    def metadata(self) -> dict:
        return {
            "loss_names": list(LOSS_NAMES),
            "model_revisions": {"convnext": getattr(self.convnext.config, "_commit_hash", None),
                                "dino": getattr(self.dino.config, "_commit_hash", None)},
            "normalization": {"convnext_mean": self.convnext_mean.flatten().tolist(),
                              "convnext_std": self.convnext_std.flatten().tolist(),
                              "dino_mean": self.dino_mean.flatten().tolist(),
                              "dino_std": self.dino_std.flatten().tolist()},
            "geometry": "paired full-image bicubic antialiased resize to 224; no random crop",
            "normalized_ab_gradient_scale": 127.5,
            "huber_ab": "SmoothL1 on physical CIELAB a*,b*, beta=5",
            "gradient_ab": "mean L1 difference of horizontal/vertical finite differences on physical a*,b*",
            "lpips_vgg": "LPIPS VGG16 on RGB resized to 224 and scaled to [-1,1]",
            "convnext_perceptual": "equal-weight spatial MSE over all four ConvNeXtV2 stages; processor mean/std; full-image resize (not exact pMF)",
            "dinov3_perceptual": "mean patch cosine distance at layers 3,6,9,12 excluding CLS/registers; processor normalization",
            "config": self.config.__dict__,
        }
