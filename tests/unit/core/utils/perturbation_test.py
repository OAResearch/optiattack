import sys

import numpy as np
import pytest

from core.utils import perturbation


def test_metrics_of_known_pair():
    orig = np.zeros((4, 5, 3), dtype=np.uint8)
    adv = orig.copy(); adv[0, 0] = [3, 0, 4]; adv[2, 1, 2] = 10
    m = perturbation.metrics(orig, adv)
    assert m["l0_pixels"] == 2
    assert m["linf"] == 10.0
    assert m["l2"] == pytest.approx(np.sqrt(9 + 16 + 100))


def test_identical_images():
    img = np.full((8, 8, 3), 7, dtype=np.uint8)
    m = perturbation.metrics(img, img)
    assert m["l0_pixels"] == 0 and m["l2"] == 0.0 and m["psnr"] == float("inf")


def test_metrics_do_not_wrap_around_on_uint8():
    orig = np.zeros((2, 2, 3), dtype=np.uint8)
    adv = np.full((2, 2, 3), 250, dtype=np.uint8)
    assert perturbation.linf(orig, adv) == 250.0
    assert perturbation.linf(adv, orig) == 250.0


def test_psnr_of_known_pair():
    orig = np.zeros((2, 2, 3), dtype=np.uint8)
    adv = np.full((2, 2, 3), 255, dtype=np.uint8)
    assert perturbation.psnr(orig, adv) == pytest.approx(0.0)


def test_ssim_uses_data_range_255():
    pytest.importorskip("skimage")
    rng = np.random.RandomState(0)
    orig = rng.randint(0, 256, (32, 32, 3)).astype(np.uint8)
    adv = np.clip(orig.astype(int) + rng.randint(-40, 41, orig.shape), 0, 255).astype(np.uint8)
    assert perturbation.ssim(orig, adv) < 0.99  # int32 without data_range would give ~1.0


def test_ssim_is_none_without_skimage(monkeypatch):
    monkeypatch.setitem(sys.modules, "skimage", None)
    monkeypatch.setitem(sys.modules, "skimage.metrics", None)
    assert perturbation.ssim(np.zeros((8, 8, 3), np.uint8), np.ones((8, 8, 3), np.uint8)) is None
