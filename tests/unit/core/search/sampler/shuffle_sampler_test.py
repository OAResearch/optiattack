"""Unit tests for the shuffle sampler."""
from collections import Counter
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.search.service.randomness import Randomness
from core.search.service.sampler.shuffle_sampler import ShuffleSampler


def build_shuffle_sampler(block_size, height=16, width=20):
    config = {"seed": 5, "image_height": height, "image_width": width, "shuffle_block_size": block_size}
    archive = MagicMock()
    archive.is_empty.return_value = True
    archive.image.array = np.random.default_rng(0).integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return ShuffleSampler(Randomness(config), archive, config), archive.image.array


@pytest.mark.parametrize("block_size", [2, 3, 4, 5])
def test_sample_creates_a_shuffled_block_of_the_configured_size(block_size):
    sampler, image = build_shuffle_sampler(block_size)

    for _ in range(20):
        actions = sampler.sample().get_actions()
        rows = sorted({a.get_location()[0] for a in actions})
        cols = sorted({a.get_location()[1] for a in actions})

        # A contiguous block_size x block_size square inside the image
        assert len(actions) == block_size * block_size
        assert rows == list(range(rows[0], rows[0] + block_size))
        assert cols == list(range(cols[0], cols[0] + block_size))
        assert 0 <= rows[0] and rows[-1] < image.shape[0]
        assert 0 <= cols[0] and cols[-1] < image.shape[1]

        # Colors are a permutation of the original block's pixels
        original = Counter(tuple(int(v) for v in image[r, c]) for r in rows for c in cols)
        shuffled = Counter(tuple(int(v) for v in a.get_color()) for a in actions)
        assert shuffled == original
