"""A small conditional DiT-style transformer for Pixel MeanFlow."""

from __future__ import annotations

import hashlib
import math
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .lab import compose_lab
from .pmf import average_velocity


class ScalarFourierEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10_000):
        super().__init__()
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32)
            / max(half, 1)
        )
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.dim = dim

    def forward(self, value: Tensor) -> Tensor:
        value = value.float().reshape(-1, 1)
        angles = value * self.frequencies.reshape(1, -1)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if embedding.shape[-1] < self.dim:
            embedding = F.pad(embedding, (0, self.dim - embedding.shape[-1]))
        return embedding


class TimeCondition(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.embedding = ScalarFourierEmbedding(dim)
        self.projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, r: Tensor, t: Tensor) -> Tensor:
        return self.projection(self.embedding(t - r))


class SelfAttention(nn.Module):
    """Explicit attention avoids fused kernels that lack forward-mode AD."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError("hidden size must be divisible by number of heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.projection = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = scores.float().softmax(dim=-1).to(dtype=q.dtype)
        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).reshape(batch, tokens, dim)
        return self.projection(attended)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attention = SelfAttention(dim, heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    @staticmethod
    def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation(condition).chunk(6, dim=-1)
        )
        x = x + gate_attn.unsqueeze(1) * self.attention(
            self._modulate(self.norm1(x), shift_attn, scale_attn)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            self._modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, dim: int, patch_size: int):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size
        )
        self.patch_size = patch_size

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x).flatten(2).transpose(1, 2)


class PMFTiny(nn.Module):
    """pMF-Tiny with clean-x and auxiliary instantaneous-velocity heads."""

    def __init__(
        self,
        *,
        resolution: Union[int, Tuple[int, int]] = 256,
        patch_size: int = 16,
        hidden_size: int = 384,
        depth: int = 12,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        in_channels: int = 3,
        state_channels: int = 2,
    ):
        super().__init__()
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        if any(size % patch_size for size in resolution):
            raise ValueError("resolution must be divisible by patch_size")
        if in_channels != 3 or state_channels != 2:
            raise ValueError("PMFTiny expects [z_a, z_b, L] and predicts [a, b]")
        self.resolution = tuple(resolution)
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.state_channels = state_channels
        self.patch_rows = resolution[0] // patch_size
        self.patch_cols = resolution[1] // patch_size
        self.patch_area = patch_size * patch_size

        self.patch_embed = PatchEmbed(in_channels, hidden_size, patch_size)
        self.time_condition = TimeCondition(hidden_size)
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, heads, mlp_ratio) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        output_dim = state_channels * self.patch_area
        self.clean_head = nn.Linear(hidden_size, output_dim)
        self.velocity_head = nn.Linear(hidden_size, output_dim)
        nn.init.normal_(self.clean_head.weight, std=0.02)
        nn.init.zeros_(self.clean_head.bias)
        nn.init.normal_(self.velocity_head.weight, std=0.02)
        nn.init.zeros_(self.velocity_head.bias)
        self.main_model_evaluations = 0
        self.last_sample_nfe = 0

    def _unpatchify(self, tokens: Tensor) -> Tensor:
        batch = tokens.shape[0]
        tokens = tokens.reshape(
            batch,
            self.patch_rows,
            self.patch_cols,
            self.patch_size,
            self.patch_size,
            self.state_channels,
        )
        tokens = tokens.permute(0, 5, 1, 3, 2, 4)
        return tokens.reshape(
            batch,
            self.state_channels,
            self.patch_rows * self.patch_size,
            self.patch_cols * self.patch_size,
        )

    def forward(
        self,
        z: Tensor,
        L: Tensor,
        r: Tensor,
        t: Tensor,
        *,
        return_velocity: bool = True,
    ):
        if z.ndim != 4 or z.shape[1] != 2:
            raise ValueError("z must be [B,2,H,W]; L is not part of the state")
        if L.ndim != 4 or L.shape[1] != 1 or L.shape[0] != z.shape[0]:
            raise ValueError("L must be [B,1,H,W] and match z")
        if z.shape[-2:] != self.resolution or L.shape[-2:] != self.resolution:
            raise ValueError(f"expected spatial resolution {self.resolution}")
        hidden = self.patch_embed(torch.cat([z, L], dim=1))
        condition = self.time_condition(r, t)
        for block in self.blocks:
            hidden = block(hidden, condition)
        hidden = self.final_norm(hidden)
        clean = self._unpatchify(self.clean_head(hidden))
        if not return_velocity:
            return clean, None
        velocity_head = self._unpatchify(self.velocity_head(hidden))
        velocity = average_velocity(z, velocity_head, t)
        return clean, velocity

    @property
    def total_training_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def auxiliary_v_head_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.velocity_head.parameters())

    @property
    def inference_required_parameters(self) -> int:
        return self.total_training_parameters - self.auxiliary_v_head_parameters

    def parameter_report(self) -> dict:
        return {
            "total_training_parameters": self.total_training_parameters,
            "inference_required_parameters": self.inference_required_parameters,
            "auxiliary_v_head_parameters": self.auxiliary_v_head_parameters,
        }

    @staticmethod
    def _stable_seed(image_id: str, seed: int) -> int:
        digest = hashlib.sha256(f"{image_id}|{seed}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="little") % (2**63 - 1)

    @torch.no_grad()
    def sample(
        self,
        L: Tensor,
        *,
        seed: Optional[int] = None,
        seeds: Optional[Sequence[int]] = None,
        image_ids: Optional[Sequence[str]] = None,
    ) -> Tensor:
        """Generate chroma in one main network evaluation.

        With a single input image, a sequence of seeds returns one sample per
        seed.  Per-example noise is generated independently on CPU so changing
        batch size or DDP rank does not change a sample's random stream.
        """

        if seed is None and seeds is None:
            seed = 0
        if seed is not None and seeds is not None:
            raise ValueError("pass either seed or seeds, not both")
        batch = L.shape[0]
        if seeds is None:
            requested_seeds = [int(seed)] * batch
        else:
            requested_seeds = [int(value) for value in seeds]
            if batch != 1 and len(requested_seeds) != batch:
                raise ValueError("seeds must match batch size unless L has batch one")
            if batch == 1:
                L = L.expand(len(requested_seeds), -1, -1, -1)
        if image_ids is None:
            ids = [str(index) for index in range(L.shape[0])]
        else:
            if len(image_ids) != batch:
                raise ValueError("image_ids must match the original L batch")
            ids = list(image_ids)
            if batch == 1 and len(requested_seeds) > 1:
                ids = ids * len(requested_seeds)
        if len(ids) != len(requested_seeds):
            raise ValueError("seed and image_id expansion did not match")

        noise = []
        for index, requested_seed in enumerate(requested_seeds):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self._stable_seed(ids[index], requested_seed))
            noise.append(
                torch.randn(
                    (1, self.state_channels, *self.resolution),
                    generator=generator,
                    dtype=torch.float32,
                )
            )
        z = torch.cat(noise, dim=0).to(device=L.device, dtype=L.dtype)
        r = torch.zeros(L.shape[0], device=L.device, dtype=L.dtype)
        t = torch.ones(L.shape[0], device=L.device, dtype=L.dtype)
        clean, _ = self(z, L, r, t, return_velocity=False)
        one_step_velocity = average_velocity(z, clean, t)
        generated = z - (t - r).reshape(-1, 1, 1, 1) * one_step_velocity
        self.last_sample_nfe = 1
        self.main_model_evaluations += 1
        return generated

    @torch.no_grad()
    def sample_lab(self, L: Tensor, **kwargs) -> Tensor:
        """Return ``[L_original, sampled_ab]`` without modifying luminance."""

        ab = self.sample(L, **kwargs)
        if ab.shape[0] != L.shape[0]:
            L = L.expand(ab.shape[0], -1, -1, -1)
        return compose_lab(L, ab)
