"""Random sampler implementation."""

from core.search.action import Action
from core.search.individual import Individual
from core.search.service.sampler.sampler import Sampler


class RandomSampler(Sampler):

    """Random sampler implementation."""

    def sample(self):
        """Sample an individual randomly."""
        individual = Individual()
        number_of_actions = self.randomness.next_int(self.config.get("min_action_size"),
                                                     self.config.get("max_action_size"))
        for _ in range(number_of_actions):
            action = self.sample_random_action()
            while not individual.add_action(action):
                action = self.sample_random_action()
        return individual

    def sample_random_action(self):
        """Return a random action."""
        location = (self.randomness.next_int(0, self.config.get("image_width")),
                    self.randomness.next_int(0, self.config.get("image_height")))
        color = self.randomness.next_int(0, 255), self.randomness.next_int(0, 255), self.randomness.next_int(0, 255)
        return Action(location, color[0], color[1], color[2])
