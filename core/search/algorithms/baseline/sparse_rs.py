"""
Numpy port of the L0 (pixel) version of Sparse-RS, used as OptiAttack's L0 black-box baseline.

Croce, Andriushchenko, Singh, Flammarion, Hein, "Sparse-RS: a versatile framework for query-efficient sparse black-box
adversarial attacks", AAAI 2022. Ported from the official code, `rs_attacks.py` of github.com/fra31/sparse-rs at
commit 21d875969a1455e4d5b26dcf32c843e6262d1f9c (`RSAttack.attack_single_run`, branch `norm == 'L0'`, and
`RSAttack.p_selection`): https://github.com/fra31/sparse-rs/blob/21d875969a1455e4d5b26dcf32c843e6262d1f9c/rs_attacks.py

Algorithm (as in the official code): k random distinct pixels are set to random corner colours of {0, 255}^C. In every
iteration n = max(int(p_selection(it) * k), 1) of them are put back to their original colour and n random unchanged
pixels get new random corner colours; the candidate replaces the current best only if its loss is strictly lower.
`p_selection` is the official piecewise-constant L0 schedule with `resc_schedule=True` (the iteration is rescaled to a
10000-iteration run relative to `max_iter`). The first prediction is the initial image, so a run makes `max_iter`
predictions, like the official `n_queries`.

Deviations from the official code:

- Loss: the margin of the NUT's probabilities instead of the logits, matching OptiAttack's fitness order
  (untargeted p[orig] - max_{j != orig} p[j], targeted max_{j != t} p[j] - p[t]). The official targeted attack uses
  the cross-entropy loss instead.
- Every newly changed pixel gets a corner colour different from its original colour, so every image has exactly k
  changed pixels and every candidate differs from the current best. The official code ensures a different colour only
  for 1-pixel updates; for larger updates a drawn corner colour may equal the original pixel (only possible when that
  pixel already has a corner colour).
- No own success stop: the official loop stops when the margin becomes negative; here OptiAttack's query tracker stops
  the attack (`baseline_stop_at_success`, `fitness_threshold`) and the budget (`BudgetExhausted`).
- Only one image, one restart, no `constant_schedule`, and random numbers come from a `np.random.RandomState` seeded
  from OptiAttack's seeded randomness (not torch), so runs are reproducible but not identical to the official ones.
"""
from typing import Optional

import numpy as np

# Official L0 schedule of p_selection: (upper bound of the rescaled iteration, divisor of p_init).
_SCHEDULE = ((0, 1), (50, 2), (200, 4), (500, 5), (1000, 6), (2000, 8), (4000, 10), (6000, 12), (8000, 15))
_LAST_DIVISOR = 20


class SparseRS:

    """Sparse-RS L0 attack with the `generate(x, y)` interface of the ART attacks used by `BaselineAlgorithm`."""

    def __init__(self, estimator, k: int, p_init: float, max_iter: int, targeted: bool, randomness_seed: int):
        """Creates the attack; k is the number of changed pixels and max_iter the length of the p schedule."""
        if k < 1:
            raise ValueError("sparse_rs_k must be at least 1 (got %d)." % k)
        if max_iter < 1:
            raise ValueError("The Sparse-RS iteration limit must be at least 1 (got %d)." % max_iter)
        self.estimator = estimator
        self.k = int(k)
        self.p_init = float(p_init)
        self.max_iter = int(max_iter)
        self.targeted = targeted
        self.randomness_seed = int(randomness_seed)

    def p_selection(self, it: int) -> float:
        """Fraction of the k pixels moved at iteration `it` (official L0 schedule, rescaled to max_iter)."""
        it = int(it / self.max_iter * 10000)
        for upper, divisor in _SCHEDULE:
            if it <= upper:
                return self.p_init / divisor
        return self.p_init / _LAST_DIVISOR

    def _loss(self, image: np.ndarray, label: int) -> float:
        """Margin loss of one HxWxC image; negative once the attack has succeeded."""
        scores = np.asarray(self.estimator.predict(image[np.newaxis].astype(np.float32)), dtype=np.float64)[0]
        others = np.array(scores, copy=True)
        others[label] = -np.inf
        if self.targeted:
            return float(others.max() - scores[label])
        return float(scores[label] - others.max())

    @staticmethod
    def _corner_colours(rng: np.random.RandomState, original: np.ndarray) -> np.ndarray:
        """Random corner colours of {0, 255}^C, each different from the original colour of its pixel."""
        colours = rng.randint(0, 2, size=original.shape).astype(np.float64) * 255.0
        same = np.all(colours == original, axis=-1)
        while same.any():
            colours[same] = rng.randint(0, 2, size=(int(same.sum()), original.shape[-1])) * 255.0
            same = np.all(colours == original, axis=-1)
        return colours

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """Attacks one image x (1xHxWxC, 0-255); y is the one-hot original class (untargeted) or target class."""
        if y is None:
            raise ValueError("Sparse-RS needs y: the original class (untargeted) or the target class (targeted).")
        height, width, channels = x.shape[1:]
        n_pixels = height * width
        if self.k >= n_pixels:
            raise ValueError("sparse_rs_k=%d must be below the number of pixels of the image (%d)."
                             % (self.k, n_pixels))
        label = int(np.argmax(y[0]))
        rng = np.random.RandomState(self.randomness_seed)
        original = np.asarray(x[0], dtype=np.float64).reshape(n_pixels, channels)

        order = rng.permutation(n_pixels)
        changed, unchanged = order[:self.k].copy(), order[self.k:].copy()
        best = original.copy()
        best[changed] = self._corner_colours(rng, original[changed])
        best_loss = self._loss(best.reshape(height, width, channels), label)

        for it in range(1, self.max_iter):
            n_moved = max(int(self.p_selection(it) * self.k), 1)
            ind_p = rng.permutation(self.k)[:n_moved]
            ind_np = rng.permutation(n_pixels - self.k)[:n_moved]
            removed, added = changed[ind_p], unchanged[ind_np]

            candidate = best.copy()
            candidate[removed] = original[removed]
            candidate[added] = self._corner_colours(rng, original[added])
            loss = self._loss(candidate.reshape(height, width, channels), label)

            if loss < best_loss:
                best, best_loss = candidate, loss
                changed[ind_p], unchanged[ind_np] = added, removed

        return best.reshape(1, height, width, channels).astype(np.float32)
