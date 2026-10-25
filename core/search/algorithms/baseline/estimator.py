"""ART classifier that forwards every prediction to the NUT through OptiAttack's remote controller."""
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
from art.estimators.classification import BlackBoxClassifierNeuralNetwork  # type: ignore[import-untyped]

from core.search.algorithms.baseline.nut_interface import to_one_hot, to_probability_vector
from core.search.algorithms.baseline.query_tracker import QueryTracker
from core.utils.nut_request import NutRequest


class RemoteNutEstimator(BlackBoxClassifierNeuralNetwork):

    """
    ART classifier whose predictions come from the NUT, so ART attacks see what OptiAttack sees.

    Images are rounded and clipped to uint8 before they are sent. A small LRU cache keyed by those bytes answers
    repeated queries of the same image (ART re-predicts images it has already scored, e.g. Square's loss on the
    current best) without sending them to the NUT or counting them. This assumes a deterministic NUT. Results seeded
    with `seed_cache` (the original image, already scored by `run_nut`) are kept for the whole attack.

    Only black-box prediction is supported for now. The white-box stage will add `loss_gradient` and
    `class_gradient` to this same class, with the gradients served by the NUT client.
    """

    def __init__(self, remote_controller, labels: List[str], input_shape: Tuple[int, int, int],
                 tracker: QueryTracker, label_only: bool = False, cache_size: int = 64):
        """Creates the estimator for a NUT with the given class labels and input shape (HxWxC)."""
        super().__init__(
            predict_fn=self._predict_batch,
            input_shape=input_shape,
            nb_classes=len(labels),
            channels_first=False,
            clip_values=(0.0, 255.0),
        )
        self.remote_controller = remote_controller
        self.labels = list(labels)
        self.tracker = tracker
        self.label_only = label_only
        self.cache_size = cache_size
        self._cache: "OrderedDict[bytes, np.ndarray]" = OrderedDict()
        self._seeded: Dict[bytes, np.ndarray] = {}

    def seed_cache(self, image: np.ndarray, result: NutRequest) -> None:
        """Permanently caches a result obtained outside the attack (e.g. of the original image); it is never counted."""
        self._seeded[self._to_uint8(image).tobytes()] = to_probability_vector(result, self.labels)

    @staticmethod
    def _to_uint8(image: np.ndarray) -> np.ndarray:
        """Rounds and clips an image to the uint8 image that is sent to the NUT."""
        return np.ascontiguousarray(np.clip(np.rint(image), 0, 255).astype(np.uint8))

    def _predict_batch(self, x: np.ndarray) -> np.ndarray:
        """Queries the NUT once per image of the batch and returns the stacked vectors."""
        return self._query_images(x)

    def _query_images(self, x: np.ndarray) -> np.ndarray:
        """Sends each image to the NUT, enforcing the budget before every single query."""
        vectors = []
        for image in x:
            vector = self._cached_or_query(self._to_uint8(image))
            vectors.append(to_one_hot(vector) if self.label_only else vector)
        return np.asarray(vectors, dtype=np.float32).reshape(len(vectors), len(self.labels))

    def _cached_or_query(self, uint8_image: np.ndarray) -> np.ndarray:
        """Returns the cached probability vector of an already queried image, otherwise queries the NUT."""
        key = uint8_image.tobytes()
        if key in self._seeded:
            self.tracker.cache_hit()
            return self._seeded[key]
        if key in self._cache:
            self.tracker.cache_hit()
            self._cache.move_to_end(key)
            return self._cache[key]
        self.tracker.before_query()
        result = self.remote_controller.new_action(uint8_image)
        self.tracker.after_query(uint8_image, result)
        vector = to_probability_vector(result, self.labels)
        self._cache[key] = vector
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return vector

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:
        """Returns the NUT's probability vectors (or one-hot labels) for a batch of images."""
        return self._query_images(np.asarray(x))
