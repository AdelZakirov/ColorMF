import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from scripts import compute_fd_stats as stats
from src.data import PaletteDataset


def test_real_rgb_uses_identical_geometry_without_lab(tmp_path, monkeypatch):
    path = tmp_path / "rgb.png"
    pixels = np.random.default_rng(42).integers(0, 256, (31, 47, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    data = PaletteDataset(paths=[str(path)], size=(16, 16), return_rgb=True)
    expected = data[0]["rgb"]
    # The dedicated real loader must not invoke LAB conversion.
    monkeypatch.setattr("src.data.rgb_to_lab", lambda _: pytest.fail("unnecessary LAB conversion"))
    torch.testing.assert_close(stats.RealRGBDataset(data)[0], expected, rtol=0, atol=0)


def test_background_moments_match_direct_statistics_and_propagate_failure(monkeypatch):
    features = torch.rand(1305, 7, 1, 1, generator=torch.Generator().manual_seed(42))
    loader = DataLoader(features, batch_size=64)
    monkeypatch.setattr(stats, "extract_fid_features", lambda _network, rgb: rgb.flatten(1))
    monkeypatch.setattr(stats, "rgb_to_lab", lambda _: (torch.zeros(1, 1, 1), torch.zeros(2, 1, 1)))
    monkeypatch.setattr(stats, "compare_rgb", lambda *_: {})
    progress = stats.Progress("test", 1e9)
    actual, _ = stats.collect_real_statistics(None, loader, torch.device("cpu"), progress, len(features))
    mean, covariance = actual.finalize()
    assert actual.count == len(features)  # Covers a full asynchronous block and tail.
    np.testing.assert_allclose(mean, features.numpy().reshape(1305, 7).mean(0, dtype=np.float64), atol=1e-14)
    np.testing.assert_allclose(covariance, np.cov(features.numpy().reshape(1305, 7), rowvar=False), atol=1e-14)
    def nonfinite_features(_network, rgb):
        result = rgb.flatten(1).clone()
        result[0, 0] = float("nan")
        return result
    monkeypatch.setattr(stats, "extract_fid_features", nonfinite_features)
    with pytest.raises(ValueError, match="finite"):
        stats.collect_real_statistics(None, loader, torch.device("cpu"), progress, len(features))
