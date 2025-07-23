"""Gaussian sampler implementation."""

from core.search.action import Action
from core.search.individual import Individual
from core.search.service.sampler.sampler import Sampler


class GaussianSampler(Sampler):

    """Gaussian sampler implementation."""

    def sample(self):
        """Sample an individual randomly."""
        individual = Individual()
        number_of_actions = self.randomness.next_int(self.config.get("min_action_size"),
                                                     self.config.get("max_action_size"))
        for _ in range(number_of_actions):
            action = self.sample_random_action()
            while not individual.add_action(action, replace=False):
                action = self.sample_random_action()
        return individual

    def sample_random_action(self):
        """Return a random action."""
        location = (self.randomness.next_int(0, self.config.get("image_height")),
                    self.randomness.next_int(0, self.config.get("image_width")))
        sigma = self.config.get("mutation_sigma")
        delta = (self.randomness.random_gaussian(0, sigma),
                 self.randomness.random_gaussian(0, sigma),
                 self.randomness.random_gaussian(0, sigma))
        color = self.archive.image.array[location] + delta
        color = self.check_limit_values(color)
        return Action(location, color[0], color[1], color[2])
