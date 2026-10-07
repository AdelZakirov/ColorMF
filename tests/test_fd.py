"""Focused FD integration invariants; no downloads or large model training."""
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from eval.metrics.fid import FeatureStatistics, frechet_distance
from src.fd_module import all_finite, ddp_backward_loss, fd_rgb, StepBatchSampler
from src.model import PixelMeanFlowB
from third_party.fd_loss.losses import (compute_frechet_distance_loss, diff_all_gather,
                                       precompute_sigma_ref_sqrt)
from third_party.fd_loss.queue import FeatureQueue


def test_sampling_equivalence_and_rgb_gradient():
    model = PixelMeanFlowB(resolution=16, patch_size=4, hidden_size=32,
                          depth=3, heads=4, aux_head_depth=1, pca_channels=8,
                          conditioning={"mode": "separate", "reinject": True}).eval()
    torch.nn.init.normal_(model.u_final_layer.linear._flax_linear.weight, std=0.01)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("v_"))
    L = torch.zeros(2, 1, 16, 16)
    noise = []
    for image_id in ["a", "b"]:
        rng = torch.Generator().manual_seed(model._stable_seed(image_id, 42))
        noise.append(torch.randn(1, 2, 16, 16, generator=rng) * 0.25)
    noise = torch.cat(noise)
    generated = model.sample_from_noise(L, noise)
    old_u, _ = model(noise, L, torch.zeros(2), torch.ones(2), return_velocity=False)
    torch.testing.assert_close(generated, noise - old_u, rtol=0, atol=0)
    torch.testing.assert_close(generated, model.sample(L, seed=42, image_ids=["a", "b"], noise_scale=.25), rtol=0, atol=0)
    fd_rgb(L, generated).square().mean().backward()
    for prefix in ["u_final_layer", "shared_blocks", "condition_embedder"]:
        grads = [p.grad for name, p in model.named_parameters() if name.startswith(prefix)]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
    assert all(p.grad is None for name, p in model.named_parameters() if name.startswith("v_"))


def test_fd_evaluator_moments_and_streaming_queue():
    rng = np.random.default_rng(42)
    real_features, generated_features = rng.normal(size=(40, 6)), rng.normal(size=(40, 6)) + 0.1
    real, generated = FeatureStatistics(), FeatureStatistics()
    real.update(real_features)
    generated.update(generated_features)
    mu_r, cov_r = map(torch.from_numpy, real.finalize())
    mu_g, cov_g = map(torch.from_numpy, generated.finalize())
    upstream = compute_frechet_distance_loss(mu_r, cov_r, mu=mu_g, sigma=cov_g,
                                             sigma_ref_sqrt=precompute_sigma_ref_sqrt(cov_r))
    np.testing.assert_allclose(upstream.item(), frechet_distance(real, generated), rtol=1e-6, atol=1e-6)
    queue = FeatureQueue(size=50000, feat_dim=6, ema_beta=.999)
    features = torch.from_numpy(generated_features).float()
    queue.accumulate_batch(features[:13])
    queue.accumulate_batch(features[13:])
    queue._finalize_streaming_init()
    torch.testing.assert_close(queue.mu_ema, features.double().mean(0))
    population_cov = queue.m2_ema - queue.mu_ema[:, None] * queue.mu_ema[None, :]
    torch.testing.assert_close(population_cov * 40 / 39, torch.cov(features.double().T))
    clone = deepcopy(queue)
    new = torch.randn(8, 6, requires_grad=True)
    mu, sigma = queue.build_feats_stats(new)
    queue.enqueue(new)
    torch.testing.assert_close(queue.mu_ema, mu.detach())
    torch.testing.assert_close(queue.m2_ema - mu[:, None] * mu[None, :], sigma)
    clone.load_state_dict(queue.state_dict())
    torch.testing.assert_close(clone.mu_ema, queue.mu_ema)


def _distributed_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        torch.manual_seed(42)
        model = torch.nn.Linear(4, 6).double()
        reference = deepcopy(model)
        ddp = DDP(model)
        x = torch.randn(12, 4, dtype=torch.float64)
        initial = torch.randn(32, 6)
        queue = FeatureQueue(size=50000, feat_dim=6, ema_beta=.999)
        queue.accumulate_batch(initial)
        queue._finalize_streaming_init()
        real_mu = torch.zeros(6, dtype=torch.float64)
        real_cov = torch.eye(6, dtype=torch.float64)
        def loss(features):
            mu, sigma = queue.build_feats_stats(features)
            raw = compute_frechet_distance_loss(real_mu, real_cov, mu=mu, sigma=sigma)
            return raw / (raw.detach() + .01)
        loss(reference(x)).backward()
        local = ddp(x[rank * 6:(rank + 1) * 6])
        ddp_backward_loss(loss(diff_all_gather(local))).backward()
        for expected, actual in zip(reference.parameters(), model.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, atol=1e-12, rtol=1e-8)
        assert not all_finite(torch.tensor(rank == 0))
    finally:
        dist.destroy_process_group()


def test_distributed_gradient_matches_global_batch(tmp_path):
    mp.spawn(_distributed_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
    whole = iter(StepBatchSampler(32, 4, 0, 1, 42))
    batches = [next(whole) for _ in range(10)]
    restored = iter(StepBatchSampler(32, 4, 0, 1, 42, start=7))
    assert [next(restored) for _ in range(3)] == batches[7:]
