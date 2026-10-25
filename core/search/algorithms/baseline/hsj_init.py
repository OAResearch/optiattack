"""Finds the adversarial start image that HopSkipJump needs, using OptiAttack's seeded randomness."""
import logging
from typing import Callable, Optional

import numpy as np

from core.search.service.randomness import Randomness
from core.utils.images import img_to_array, read_image, resize_image

logger = logging.getLogger(__name__)


def find_hsj_initial_image(estimator, original: np.ndarray, randomness: Randomness, config: dict,
                           is_adversarial: Callable[[np.ndarray], bool]) -> Optional[np.ndarray]:
    """Returns a float32 HxWxC adversarial image in 0-255 for HopSkipJump, or None if none was found."""
    path = config.get("hsj_init_image")
    if path:
        image = _load_init_image(str(path), config)
        if image is not None:
            if is_adversarial(estimator.predict(image[None, ...])[0]):
                return image
            logger.warning("The hsj_init_image '%s' is not adversarial for the NUT; searching a random start image.",
                           path)
    return _random_init_image(estimator, original.shape, randomness, int(config["hsj_init_size"]), is_adversarial)


def _load_init_image(path: str, config: dict) -> Optional[np.ndarray]:
    """Loads the given start image resized to the NUT input size, or warns and returns None if it cannot be read."""
    try:
        image = resize_image(read_image(path), int(config["image_width"]), int(config["image_height"]))
    except (OSError, ValueError) as error:
        logger.warning("Could not load hsj_init_image '%s' (%s); searching a random start image.", path, error)
        return None
    return img_to_array(image).astype(np.float32)


def _random_init_image(estimator, shape, randomness: Randomness, tries: int,
                       is_adversarial: Callable[[np.ndarray], bool]) -> Optional[np.ndarray]:
    """Queries up to `tries` uniform random images and returns the first adversarial one."""
    for _ in range(tries):
        candidate = np.rint(randomness.random.uniform(0, 255, size=shape)).astype(np.float32)
        if is_adversarial(estimator.predict(candidate[None, ...])[0]):
            return candidate
    return None
