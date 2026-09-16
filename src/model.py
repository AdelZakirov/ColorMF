"""Official pMF-B architecture adapted to conditional LAB colorization.

The transformer follows Lyy-iiis/pMF's PyTorch inference branch at commit
990e81a84249dbd68a128accef67eb95621d10b1. The default concat mode uses a
three-channel spatial input ``[z_ab, L]`` and two-channel ``ab`` heads;
separate mode gives state and luminance their own spatial representations.
"""

from __future__ import annotations

import hashlib
import math
from functools import partial
from typing import Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .lab import compose_lab


class TorchLinear(nn.Module):
    """Linear layer with the initialization used by the official Flax port."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_init: str = "scaled_variance", init_constant: float = 1.0,
                 bias_init: str = "zeros"):
        super().__init__()
        if weight_init == "scaled_variance":
            initializer = partial(nn.init.normal_, std=init_constant / math.sqrt(in_features))
        elif weight_init == "zeros":
            initializer = nn.init.zeros_
        else:
            raise ValueError(f"invalid weight_init: {weight_init}")
        if bias_init != "zeros":
            raise ValueError(f"invalid bias_init: {bias_init}")
        self._flax_linear = nn.Linear(in_features, out_features, bias=bias)
        initializer(self._flax_linear.weight)
        if bias:
            nn.init.zeros_(self._flax_linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self._flax_linear(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        output = x * torch.rsqrt(torch.square(x).mean(dim=-1, keepdim=True) + self.eps)
        return output.to(x.dtype) * self.weight


class SwiGLUMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int,
                 weight_init: str = "scaled_variance", weight_init_constant: float = 1.0):
        super().__init__()
        kwargs = dict(bias=False, weight_init=weight_init, init_constant=weight_init_constant)
        self.w1 = TorchLinear(in_features, hidden_features, **kwargs)
        self.w3 = TorchLinear(in_features, hidden_features, **kwargs)
        self.w2 = TorchLinear(hidden_features, in_features, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256,
                 weight_init: str = "scaled_variance", init_constant: float = 1.0):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        kwargs = dict(out_features=hidden_size, bias=True, weight_init=weight_init,
                      init_constant=init_constant, bias_init="zeros")
        self.mlp = nn.Sequential(
            TorchLinear(frequency_embedding_size, **kwargs), nn.SiLU(),
            TorchLinear(hidden_size, **kwargs),
        )

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
        half = dim // 2
        frequencies = torch.exp(-math.log(max_period) * torch.arange(
            half, dtype=torch.float32, device=t.device) / half)
        angles = t[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class BottleneckPatchEmbedder(nn.Module):
    """Two-stage patch embedder from official pMF."""

    def __init__(self, input_size: int, initial_patch_size: int, pca_channels: int,
                 in_channels: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.patch_size = (initial_patch_size, initial_patch_size)
        self.grid_size = tuple(size // initial_patch_size for size in (input_size, input_size))
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj1 = nn.Conv2d(in_channels, pca_channels, kernel_size=self.patch_size,
                               stride=self.patch_size, bias=bias)
        self.proj2 = nn.Conv2d(pca_channels, hidden_size, kernel_size=1, bias=bias)
        fan_in = initial_patch_size * initial_patch_size * in_channels
        limit1 = math.sqrt(6.0 / (fan_in + pca_channels))
        limit2 = math.sqrt(6.0 / (pca_channels + hidden_size))
        nn.init.uniform_(self.proj1.weight, -limit1, limit1)
        nn.init.uniform_(self.proj2.weight, -limit2, limit2)
        if bias:
            nn.init.zeros_(self.proj1.bias)
            nn.init.zeros_(self.proj2.bias)

    def forward(self, x: Tensor) -> Tensor:
        batch, _, height, width = x.shape
        if height != self.input_size or width != self.input_size:
            raise ValueError(f"expected {self.input_size}x{self.input_size}, got {height}x{width}")
        x = self.proj2(self.proj1(x))
        return x.permute(0, 2, 3, 1).reshape(batch, -1, x.shape[1])


def precompute_rope_freqs(dim: int, seq_len: int, theta: float = 10_000.0) -> Tensor:
    rotary_dim = dim // 2
    side = math.isqrt(seq_len)
    if side * side != seq_len:
        raise ValueError("RoPE requires a square spatial token grid")
    frequencies = 1.0 / (theta ** (
        torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    positions = torch.arange(side, dtype=torch.float32)
    axis = torch.einsum("i,j->ij", positions, frequencies)
    grid = torch.cat([axis[:, None, :].tile(1, side, 1),
                      axis[None, :, :].tile(side, 1, 1)], dim=-1)
    return torch.complex(torch.cos(grid).reshape(seq_len, rotary_dim),
                         torch.sin(grid).reshape(seq_len, rotary_dim))


def apply_rotary_pos_emb(x: Tensor, rope_freqs: Tensor) -> Tensor:
    """Apply 2-D RoPE only to trailing spatial tokens; preserve time tokens."""
    spatial_tokens = rope_freqs.shape[0]
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2).contiguous())
    prefix = x_complex[:, :-spatial_tokens]
    spatial = x_complex[:, -spatial_tokens:] * rope_freqs[None, :, None, :]
    return torch.view_as_real(torch.cat([prefix, spatial], dim=1)).flatten(-2).to(x.dtype)


class RoPEAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int,
                 weight_init: str = "scaled_variance", weight_init_constant: float = 1.0):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        kwargs = dict(in_features=hidden_size, out_features=hidden_size, bias=False,
                      weight_init=weight_init, init_constant=weight_init_constant)
        self.q_proj = TorchLinear(**kwargs)
        self.k_proj = TorchLinear(**kwargs)
        self.v_proj = TorchLinear(**kwargs)
        self.out_proj = TorchLinear(**kwargs)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x: Tensor, rope_freqs: Tensor) -> Tensor:
        batch, sequence, _ = x.shape
        shape = (batch, sequence, self.num_heads, self.head_dim)
        q = apply_rotary_pos_emb(self.q_norm(self.q_proj(x).reshape(shape)), rope_freqs)
        k = apply_rotary_pos_emb(self.k_norm(self.k_proj(x).reshape(shape)), rope_freqs)
        v = self.v_proj(x).reshape(shape)
        weights = torch.einsum("bqhd,bkhd->bhqk", q / math.sqrt(self.head_dim), k)
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(v.dtype)
        attended = torch.einsum("bhqk,bkhd->bqhd", weights, v)
        return self.out_proj(attended.reshape(batch, sequence, self.hidden_size))


class TransformerBlock(nn.Module):
    """Official pMF block with zero-initialized vector residual gates."""
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 8 / 3,
                 weight_init: str = "scaled_variance", weight_init_constant: float = 1.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = RoPEAttention(hidden_size, num_heads, weight_init, weight_init_constant)
        self.norm2 = RMSNorm(hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        if hidden_size > 1024:
            mlp_hidden = (mlp_hidden + 7) // 8 * 8
        self.mlp = SwiGLUMlp(hidden_size, mlp_hidden, weight_init, weight_init_constant)
        self.attn_scale = nn.Parameter(torch.zeros(hidden_size))
        self.mlp_scale = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: Tensor, rope_freqs: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), rope_freqs) * self.attn_scale
        return x + self.mlp(self.norm2(x)) * self.mlp_scale


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = TorchLinear(hidden_size, patch_size * patch_size * out_channels,
                                  bias=True, weight_init="zeros", bias_init="zeros")

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.norm(x))


class PixelMeanFlowB(nn.Module):
    """pMF-B adapted from RGB generation to conditional ``p(ab | L)``.

    ``conditioning.mode='concat'`` is the legacy architecture; ``'separate'``
    uses independent state and luminance token representations.
    """

    def __init__(self, *, resolution: Union[int, Tuple[int, int]] = 256,
                 patch_size: int = 16, in_channels: int = 3, out_channels: int = 2,
                 hidden_size: int = 768, depth: int = 16, num_heads: int = 12,
                 heads: Optional[int] = None, mlp_ratio: float = 8 / 3,
                 aux_head_depth: int = 8, pca_channels: int = 128,
                 num_time_tokens: int = 4, token_init_constant: float = 1.0,
                 embedding_init_constant: float = 1.0,
                 weight_init_constant: float = 0.32,
                 conditioning: Optional[Mapping[str, object]] = None):
        super().__init__()
        if heads is not None:
            num_heads = heads
        if isinstance(resolution, (tuple, list)):
            if resolution[0] != resolution[1]:
                raise ValueError("official pMF RoPE requires square inputs")
            resolution = resolution[0]
        if resolution % patch_size:
            raise ValueError("resolution must be divisible by patch_size")
        if in_channels != 3 or out_channels != 2:
            raise ValueError("conditional pMF expects [z_ab,L] input and ab output")
        if not 0 < aux_head_depth < depth:
            raise ValueError("aux_head_depth must leave at least one shared block")
        if conditioning is None:
            conditioning = {}
        if not isinstance(conditioning, Mapping):
            raise TypeError("conditioning must be a mapping with mode and reinject")
        conditioning_mode = conditioning.get("mode", "concat")
        if conditioning_mode not in {"concat", "separate"}:
            raise ValueError("conditioning.mode must be 'concat' or 'separate'")
        conditioning_reinject = bool(conditioning.get("reinject", True))
        self.resolution = (resolution, resolution)
        self.input_size = resolution
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.state_channels = out_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.aux_head_depth = aux_head_depth
        self.num_time_tokens = num_time_tokens
        self.conditioning_mode = conditioning_mode
        self.conditioning_reinject = conditioning_reinject
        # Keep this alias convenient for callers that use the config spelling.
        self.reinject = conditioning_reinject
        if conditioning_mode == "concat":
            # This is intentionally the legacy construction path. In particular,
            # keep the module name, type, arguments, and initialization unchanged
            # so old concat checkpoints retain their exact state-dict keys.
            self.x_embedder = BottleneckPatchEmbedder(
                resolution, patch_size, pca_channels, in_channels, hidden_size, bias=True)
            self._spatial_num_patches = self.x_embedder.num_patches
            self._spatial_grid_size = self.x_embedder.grid_size
        else:
            self.state_embedder = BottleneckPatchEmbedder(
                resolution, patch_size, pca_channels, self.state_channels, hidden_size,
                bias=True)
            self.condition_embedder = BottleneckPatchEmbedder(
                resolution, patch_size, pca_channels, 1, hidden_size, bias=True)
            if (self.state_embedder.patch_size != self.condition_embedder.patch_size or
                    self.state_embedder.grid_size != self.condition_embedder.grid_size or
                    self.state_embedder.num_patches != self.condition_embedder.num_patches):
                raise ValueError(
                    "state and condition embedders must use the same patch and token grid")
            self._spatial_num_patches = self.state_embedder.num_patches
            self._spatial_grid_size = self.state_embedder.grid_size
            self.condition_norm = RMSNorm(hidden_size)
            self.condition_proj = TorchLinear(
                hidden_size, hidden_size, bias=False,
                weight_init="scaled_variance", init_constant=embedding_init_constant)
        self.h_embedder = TimestepEmbedder(
            hidden_size, weight_init="scaled_variance", init_constant=embedding_init_constant)
        token_initializer = partial(nn.init.normal_,
                                    std=token_init_constant / math.sqrt(hidden_size))
        self.time_tokens = nn.Parameter(token_initializer(
            torch.empty(1, num_time_tokens, hidden_size)))
        self.prefix_tokens = num_time_tokens
        self.pos_embed = nn.Parameter(nn.init.normal_(torch.empty(
            1, self._spatial_num_patches + num_time_tokens, hidden_size), std=0.02))
        self.register_buffer("rope_freqs", precompute_rope_freqs(
            hidden_size // num_heads, self._spatial_num_patches), persistent=False)
        block_kwargs = dict(hidden_size=hidden_size, num_heads=num_heads,
                            mlp_ratio=mlp_ratio, weight_init="scaled_variance",
                            weight_init_constant=weight_init_constant)
        self.shared_blocks = nn.ModuleList([TransformerBlock(**block_kwargs)
                                            for _ in range(depth - aux_head_depth)])
        self.u_blocks = nn.ModuleList([TransformerBlock(**block_kwargs)
                                       for _ in range(aux_head_depth)])
        self.v_blocks = nn.ModuleList([TransformerBlock(**block_kwargs)
                                       for _ in range(aux_head_depth)])
        if conditioning_mode == "separate" and conditioning_reinject:
            self.shared_condition_gates = nn.ParameterList([
                nn.Parameter(torch.zeros(hidden_size)) for _ in self.shared_blocks
            ])
            self.u_condition_gates = nn.ParameterList([
                nn.Parameter(torch.zeros(hidden_size)) for _ in self.u_blocks
            ])
            self.v_condition_gates = nn.ParameterList([
                nn.Parameter(torch.zeros(hidden_size)) for _ in self.v_blocks
            ])
        else:
            # No unused gate parameters in concat mode or when reinjection is off.
            self.shared_condition_gates = None
            self.u_condition_gates = None
            self.v_condition_gates = None
        self.u_final_layer = FinalLayer(hidden_size, patch_size, out_channels)
        self.v_final_layer = FinalLayer(hidden_size, patch_size, out_channels)
        self.main_model_evaluations = 0
        self.last_sample_nfe = 0

    @property
    def num_spatial_tokens(self) -> int:
        return self._spatial_num_patches

    def _unpatchify(self, tokens: Tensor) -> Tensor:
        side = math.isqrt(tokens.shape[1])
        if side * side != tokens.shape[1]:
            raise ValueError("spatial token count must be square")
        p = self.patch_size
        tokens = tokens.reshape(tokens.shape[0], side, side, p, p, self.out_channels)
        tokens = torch.einsum("nhwpqc->nchpwq", tokens)
        return tokens.reshape(tokens.shape[0], self.out_channels, side * p, side * p)

    @staticmethod
    def _velocity(z: Tensor, clean: Tensor, t: Tensor) -> Tensor:
        value = t.float()
        denominator = torch.minimum(torch.maximum(value, value.new_tensor(0.05)),
                                    value.new_tensor(1.0)).reshape(-1, 1, 1, 1)
        return (z.float() - clean.float()) / denominator

    def _sequence(self, z: Tensor, L: Tensor, h: Tensor) -> tuple[Tensor, Optional[Tensor]]:
        if z.ndim != 4 or z.shape[1] != 2:
            raise ValueError("z must be [B,2,H,W]; L is not stochastic state")
        if L.shape != (z.shape[0], 1, *z.shape[-2:]):
            raise ValueError("L must be [B,1,H,W] and match z")
        if z.shape[-2:] != self.resolution:
            raise ValueError(f"expected resolution {self.resolution}")
        if self.conditioning_mode == "concat":
            spatial = self.x_embedder(torch.cat([z, L], dim=1))
            condition_sequence = None
        else:
            state_tokens = self.state_embedder(z)
            condition_tokens = self.condition_embedder(L)
            projected_condition = self.condition_proj(self.condition_norm(condition_tokens))
            spatial = state_tokens + projected_condition
            condition_sequence = torch.cat([
                projected_condition.new_zeros(
                    projected_condition.shape[0], self.prefix_tokens, self.hidden_size),
                projected_condition,
            ], dim=1)
        time = self.time_tokens + self.h_embedder(h)[:, None]
        sequence = torch.cat([time, spatial], dim=1) + self.pos_embed
        return sequence, condition_sequence

    def run_blocks(
        self,
        sequence: Tensor,
        condition_sequence: Optional[Tensor],
        blocks: nn.ModuleList,
        gates: Optional[nn.ParameterList],
    ) -> Tensor:
        """Run a stack with the same optional spatial conditioning semantics."""
        if condition_sequence is not None and gates is not None:
            if len(gates) != len(blocks):
                raise ValueError("conditioning gate count must match block count")
            for block, gate in zip(blocks, gates):
                sequence = sequence + gate * condition_sequence
                sequence = block(sequence, self.rope_freqs)
            return sequence
        for block in blocks:
            sequence = block(sequence, self.rope_freqs)
        return sequence

    def forward(self, z: Tensor, L: Tensor, r: Tensor, t: Tensor, *,
                return_velocity: bool = True) -> tuple[Tensor, Optional[Tensor]]:
        sequence, condition_sequence = self._sequence(z, L, t - r)
        sequence = self.run_blocks(
            sequence, condition_sequence, self.shared_blocks, self.shared_condition_gates)
        u_sequence = self.run_blocks(
            sequence, condition_sequence, self.u_blocks, self.u_condition_gates)
        u_clean = self._unpatchify(self.u_final_layer(
            u_sequence[:, self.prefix_tokens:]))
        u = self._velocity(z, u_clean, t)
        if not return_velocity:
            return u, None
        v_sequence = self.run_blocks(
            sequence, condition_sequence, self.v_blocks, self.v_condition_gates)
        v_clean = self._unpatchify(self.v_final_layer(
            v_sequence[:, self.prefix_tokens:]))
        return u, self._velocity(z, v_clean, t)

    def auxiliary_direction(self, z: Tensor, L: Tensor, t: Tensor) -> Tensor:
        sequence, condition_sequence = self._sequence(z, L, torch.zeros_like(t))
        sequence = self.run_blocks(
            sequence, condition_sequence, self.shared_blocks, self.shared_condition_gates)
        sequence = self.run_blocks(
            sequence, condition_sequence, self.v_blocks, self.v_condition_gates)
        clean = self._unpatchify(self.v_final_layer(
            sequence[:, self.prefix_tokens:]))
        return self._velocity(z, clean, t)

    @staticmethod
    def _stable_seed(image_id: str, seed: int) -> int:
        digest = hashlib.sha256(f"{image_id}|{seed}".encode()).digest()
        return int.from_bytes(digest[:8], "little") % (2**63 - 1)

    @torch.no_grad()
    def sample(self, L: Tensor, *, seed: Optional[int] = None,
               seeds: Optional[Sequence[int]] = None,
               image_ids: Optional[Sequence[str]] = None,
               noise_scale: float = 1.0) -> Tensor:
        if not math.isfinite(float(noise_scale)) or noise_scale <= 0:
            raise ValueError("noise_scale must be a finite positive number")
        if seed is None and seeds is None:
            seed = 0
        if seed is not None and seeds is not None:
            raise ValueError("pass either seed or seeds, not both")
        original_batch = L.shape[0]
        requested = [int(seed)] * original_batch if seeds is None else [int(x) for x in seeds]
        if original_batch == 1 and len(requested) > 1:
            L = L.expand(len(requested), -1, -1, -1)
        elif len(requested) != original_batch:
            raise ValueError("seeds must match batch size unless L has batch one")
        ids = [str(i) for i in range(original_batch)] if image_ids is None else list(image_ids)
        if len(ids) != original_batch:
            raise ValueError("image_ids must match the original L batch")
        if original_batch == 1 and len(requested) > 1:
            ids *= len(requested)
        noise = []
        for image_id, requested_seed in zip(ids, requested):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self._stable_seed(image_id, requested_seed))
            noise.append(noise_scale * torch.randn(
                (1, 2, *self.resolution), generator=generator
            ))
        z_t = torch.cat(noise).to(device=L.device, dtype=L.dtype)
        t = torch.ones(z_t.shape[0], device=L.device, dtype=L.dtype)
        r = torch.zeros_like(t)
        u, _ = self(z_t, L, r, t, return_velocity=False)
        self.last_sample_nfe = 1
        self.main_model_evaluations += 1
        return z_t - (t - r).reshape(-1, 1, 1, 1) * u

    @torch.no_grad()
    def sample_lab(self, L: Tensor, **kwargs) -> Tensor:
        ab = self.sample(L, **kwargs)
        if ab.shape[0] != L.shape[0]:
            L = L.expand(ab.shape[0], -1, -1, -1)
        return compose_lab(L, ab)

    @staticmethod
    def _count(modules) -> int:
        return sum(parameter.numel() for module in modules
                   for parameter in module.parameters())

    def parameter_report(self) -> dict[str, int]:
        embedding_modules = (
            [self.x_embedder] if self.conditioning_mode == "concat" else
            [self.state_embedder, self.condition_embedder,
             self.condition_norm, self.condition_proj]
        )
        shared = self._count([*embedding_modules, self.h_embedder, self.shared_blocks])
        shared += self.time_tokens.numel() + self.pos_embed.numel()
        if self.shared_condition_gates is not None:
            shared += self._count([self.shared_condition_gates])
        u = self._count([self.u_blocks, self.u_final_layer])
        if self.u_condition_gates is not None:
            u += self._count([self.u_condition_gates])
        v = self._count([self.v_blocks, self.v_final_layer])
        if self.v_condition_gates is not None:
            v += self._count([self.v_condition_gates])
        return {"total_training_parameters": sum(p.numel() for p in self.parameters()),
                "shared_parameters": shared, "u_head_parameters": u,
                "v_head_parameters": v, "inference_required_parameters": shared + u}
