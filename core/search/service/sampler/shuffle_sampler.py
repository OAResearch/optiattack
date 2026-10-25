"""Shuffle sampler implementation - creates blocks samples for ShuffleMutator."""

from core.search.service.sampler.sampler import Sampler
from core.search.service.shuffle_block import ShuffleBlockMixin


class ShuffleSampler(ShuffleBlockMixin, Sampler):

    """Shuffle sampler that creates block samples compatible with ShuffleMutator (same shuffle_block_size)."""

    def sample(self):
        """Sample an individual with a shuffled block."""
        return self.sample_shuffled_block(self.archive)
