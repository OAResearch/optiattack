"""Perturbation metrics between an original image and an adversarial image (HxWxC, 0-255 scale)."""
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DATA_RANGE = 255.0
_ssim_warned = False


def _difference(orig: np.ndarray, adv: np.ndarray) -> np.ndarray:
    """Return the signed difference as float64, so that uint8 arithmetic cannot wrap around."""
    return np.asarray(adv, dtype=np.float64) - np.asarray(orig, dtype=np.float64)


def l0_pixels(orig: np.ndarray, adv: np.ndarray) -> int:
    """Count the pixels where at least one channel differs."""
    return int(np.count_nonzero(np.any(_difference(orig, adv) != 0, axis=-1)))


def l2(orig: np.ndarray, adv: np.ndarray) -> float:
    """Return the Euclidean distance over all channels."""
    return float(np.sqrt(np.sum(_difference(orig, adv) ** 2)))


def linf(orig: np.ndarray, adv: np.ndarray) -> float:
    """Return the largest absolute channel difference."""
    diff = _difference(orig, adv)
    return float(np.max(np.abs(diff))) if diff.size else 0.0


def ssim(orig: np.ndarray, adv: np.ndarray) -> Optional[float]:
    """Return the SSIM (data_range=255), or None when scikit-image is not installed."""
    global _ssim_warned
    try:
        from skimage.metrics import structural_similarity  # type: ignore[import-not-found,import-untyped,unused-ignore]
    except ImportError:
        if not _ssim_warned:
            _ssim_warned = True
            logger.warning("scikit-image is not installed; the ssim column stays empty. "
                           "Install requirements-experiments.txt to compute it.")
        return None
    smaller_side = min(np.shape(orig)[0], np.shape(orig)[1])
    if smaller_side < 3:
        return None
    win_size = min(7, smaller_side if smaller_side % 2 else smaller_side - 1)  # the default window needs >= 7 pixels
    return float(structural_similarity(np.asarray(orig, dtype=np.float64), np.asarray(adv, dtype=np.float64),
                                       channel_axis=-1, data_range=_DATA_RANGE, win_size=win_size))


def psnr(orig: np.ndarray, adv: np.ndarray) -> Optional[float]:
    """Return the PSNR in dB (data_range=255); infinity for identical images."""
    mse = float(np.mean(_difference(orig, adv) ** 2))
    if mse == 0:
        return float('inf')
    return float(10 * np.log10(_DATA_RANGE ** 2 / mse))


def metrics_keys() -> List[str]:
    """Return the names of the metrics, in output order."""
    return ['l0_pixels', 'l2', 'linf', 'ssim', 'psnr']


def metrics(orig: np.ndarray, adv: np.ndarray) -> Dict[str, Optional[float]]:
    """Return all perturbation metrics of the pair."""
    return {
        'l0_pixels': l0_pixels(orig, adv),
        'l2': l2(orig, adv),
        'linf': linf(orig, adv),
        'ssim': ssim(orig, adv),
        'psnr': psnr(orig, adv),
    }
