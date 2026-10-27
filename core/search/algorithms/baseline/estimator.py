"""ART classifier that forwards every prediction to the NUT through OptiAttack's remote controller."""
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from art.estimators.classification import BlackBoxClassifierNeuralNetwork  # type: ignore[import-untyped]
from art.estimators.classification.classifier import ClassGradientsMixin  # type: ignore[import-untyped]
from art.estimators.estimator import LossGradientsMixin  # type: ignore[import-untyped]

from core.search.algorithms.baseline.nut_interface import (InvalidGradientError, to_logit_vector, to_one_hot,
                                                           to_probability_vector)
from core.search.algorithms.baseline.query_tracker import QueryTracker
from core.utils.nut_request import NutRequest


class RemoteNutEstimator(ClassGradientsMixin, LossGradientsMixin, BlackBoxClassifierNeuralNetwork):

    """
    ART classifier whose predictions come from the NUT, so ART attacks see what OptiAttack sees.

    Images are rounded and clipped to uint8 before they are sent. A small LRU cache keyed by those bytes answers
    repeated queries of the same image (ART re-predicts images it has already scored, e.g. Square's loss on the
    current best) without sending them to the NUT or counting them. This assumes a deterministic NUT. Results seeded
    with `seed_cache` (the original image, already scored by `run_nut`) are kept for the whole attack.

    For white-box attacks (`allow_gradients=True`; otherwise they raise RuntimeError, so black-box baselines can never
    query gradients) `loss_gradient` and `class_gradient` ask the NUT for gradients at the float32 image (not
    rounded). Each NUT gradient call goes through the tracker's budget check, counts as one query of the shared budget
    and is cached by (float32 image bytes, label, kind); like predictions, cache hits count towards stall detection and
    real queries reset it. With `use_logits`, `predict` returns the NUT's logits so that
    an attack's loss and its class gradients come from the same quantity; fitness and success are still computed by
    the tracker from the NUT's probability `predictions`.
    """

    def __init__(self, remote_controller, labels: List[str], input_shape: Tuple[int, int, int],
                 tracker: QueryTracker, label_only: bool = False, cache_size: int = 64, use_logits: bool = False,
                 allow_gradients: bool = False):
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
        self.use_logits = use_logits
        self.allow_gradients = allow_gradients
        self._cache: "OrderedDict[bytes, np.ndarray]" = OrderedDict()
        self._seeded: Dict[bytes, np.ndarray] = {}
        self._gradient_cache: "OrderedDict[Tuple[bytes, str, str], np.ndarray]" = OrderedDict()
        # The last image the attack asked a prediction or gradient for (e.g. the iterate the budget stopped).
        self.last_input: Optional[np.ndarray] = None

    def seed_cache(self, image: np.ndarray, result: NutRequest) -> None:
        """Permanently caches a result obtained outside the attack (e.g. of the original image); it is never counted."""
        self._seeded[self._to_uint8(image).tobytes()] = self._to_vector(result)

    def _to_vector(self, result: NutRequest) -> np.ndarray:
        """Converts a NUT result to the vector `predict` returns: logits in logits mode, otherwise probabilities."""
        if self.use_logits:
            return to_logit_vector(result, self.labels)
        return to_probability_vector(result, self.labels)

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
            self.last_input = np.array(image, dtype=np.float32, copy=True)
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
        vector = self._to_vector(result)
        self._cache[key] = vector
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return vector

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:
        """Returns the NUT's probability vectors (logits in logits mode, or one-hot labels) for a batch of images."""
        return self._query_images(np.asarray(x))

    def loss_gradient(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """Gradient of the NUT's cross-entropy loss w.r.t. each image for its label in `y` (one-hot or indices)."""
        self._check_gradients_allowed()
        x = self._check_batch(x)
        indices = self._loss_labels(y, len(x))
        grads = [self._gradient(image, index, "loss") for image, index in zip(x, indices)]
        return np.asarray(grads, dtype=np.float32).reshape(x.shape)

    def class_gradient(self, x: np.ndarray, label: Union[int, List[int], np.ndarray, None] = None,
                       **kwargs) -> np.ndarray:
        """Logit gradients w.r.t. `x`: (N, 1, H, W, C) for an int or per-sample label, (N, classes, ...) for None."""
        self._check_gradients_allowed()
        x = self._check_batch(x)
        per_sample = self._class_labels(label, len(x))
        grads = []
        for i, image in enumerate(x):
            indices = range(self.nb_classes) if per_sample is None else [per_sample[i]]
            grads.append([self._gradient(image, index, "class") for index in indices])
        n_rows = self.nb_classes if per_sample is None else 1
        return np.asarray(grads, dtype=np.float32).reshape((len(x), n_rows) + tuple(self.input_shape))

    def _check_gradients_allowed(self) -> None:
        """Raises unless the estimator was built for a white-box attack (black-box baselines never see gradients)."""
        if not self.allow_gradients:
            raise RuntimeError("Gradient queries are only allowed for white-box attacks (algorithm 'whitebox'); this "
                               "estimator was built for a black-box baseline.")

    def _check_batch(self, x: np.ndarray) -> np.ndarray:
        """Returns `x` as a float32 batch of images of the estimator's input shape."""
        x = np.asarray(x, dtype=np.float32)
        if x.shape[1:] != tuple(self.input_shape):
            raise ValueError("Expected a batch of images of shape %s, got %s." % (tuple(self.input_shape), x.shape))
        return x

    def _check_index(self, index) -> int:
        """Validates a class index and returns it as an int."""
        if not 0 <= int(index) < self.nb_classes:
            raise ValueError("Class index %s is outside [0, %d)." % (index, self.nb_classes))
        return int(index)

    def _loss_labels(self, y: np.ndarray, n_samples: int) -> List[int]:
        """Turns ART's `y` (one-hot of shape (N, classes) or indices of shape (N,) / (N, 1)) into class indices."""
        y = np.asarray(y)
        indices = np.argmax(y, axis=1) if y.ndim == 2 and y.shape[1] > 1 else y.reshape(-1)
        if len(indices) != n_samples:
            raise ValueError("Got %d labels for %d images." % (len(indices), n_samples))
        return [self._check_index(index) for index in indices]

    def _class_labels(self, label: Union[int, List[int], np.ndarray, None], n_samples: int) -> Optional[List[int]]:
        """Turns ART's `label` (int for all samples, one per sample, None for all classes) into per-sample indices."""
        if label is None:
            return None
        if isinstance(label, (int, np.integer)):
            return [self._check_index(label)] * n_samples
        indices = np.asarray(label).reshape(-1)
        if len(indices) != n_samples:
            raise ValueError("Got %d labels for %d images." % (len(indices), n_samples))
        return [self._check_index(index) for index in indices]

    def _gradient(self, image: np.ndarray, index: int, kind: str) -> np.ndarray:
        """Returns the validated NUT gradient for one float32 image and class, from the cache when possible."""
        name = self.labels[index]
        self.last_input = np.array(image, dtype=np.float32, copy=True)
        key = (self.last_input.tobytes(), name, kind)
        if key in self._gradient_cache:
            self.tracker.cache_hit()
            self._gradient_cache.move_to_end(key)
            return self._gradient_cache[key]
        self.tracker.before_query()
        raw = self.remote_controller.gradient(image, name, kind)
        self.tracker.after_gradient_query()
        gradient = self._validate_gradient(raw, name, kind)
        self._gradient_cache[key] = gradient
        while len(self._gradient_cache) > self.cache_size:
            self._gradient_cache.popitem(last=False)
        return gradient

    def _validate_gradient(self, gradient, label: str, kind: str) -> np.ndarray:
        """Raises InvalidGradientError for a gradient with the wrong shape or non-finite values."""
        gradient = np.asarray(gradient, dtype=np.float32)
        expected = tuple(self.input_shape)
        if gradient.shape != expected:
            raise InvalidGradientError(
                "The NUT returned a '%s' gradient for label '%s' with shape %s, expected the image shape %s."
                % (kind, label, gradient.shape, expected))
        bad = int(np.count_nonzero(~np.isfinite(gradient)))
        if bad:
            raise InvalidGradientError(
                "The NUT returned a '%s' gradient for label '%s' (shape %s) with %d non-finite (NaN/inf) values."
                % (kind, label, gradient.shape, bad))
        return gradient
