"""Gradient oracle: pixel saliency from a gradient-capable NUT, for future gradient-guided OptiAttack algorithms."""
from typing import Optional

import numpy as np

from core.remote.remote_controller import RemoteController
from core.search.service.archive import Archive


class GradientOracle:

    """Gives gradient-based pixel saliency when `use_gradients` is on and the NUT declares "gradients"."""

    def __init__(self, config: dict, remote_controller: RemoteController, archive: Archive) -> None:
        """Initialize the oracle; the NUT info is fetched lazily on the first availability check."""
        self.use_gradients = bool(config.get("use_gradients", False))
        self.remote_controller = remote_controller
        self.archive = archive
        self._nut_has_gradients: Optional[bool] = None

    def available(self) -> bool:
        """True only if `use_gradients` is on and the NUT info lists the "gradients" capability."""
        if not self.use_gradients:
            return False
        if self._nut_has_gradients is None:
            nut_info = self.remote_controller.get_nut_info() or {}
            self._nut_has_gradients = "gradients" in (nut_info.get("capabilities") or [])
        return self._nut_has_gradients

    def pixel_saliency(self, image: np.ndarray) -> np.ndarray:
        """Return an HxW float32 saliency map: channel-wise L2 norm of the loss gradient (one gradient query)."""
        if not self.available():
            raise RuntimeError("The gradient oracle is not available: it needs --use_gradients true and a NUT "
                               "that declares the 'gradients' capability.")
        label = self.archive.get_original_prediction_results().max_score.label
        gradient = self.remote_controller.gradient(image, label, "loss")
        return np.linalg.norm(gradient.astype(np.float32), axis=-1).astype(np.float32)
