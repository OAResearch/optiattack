"""Standard mutator module."""
from core.search.service.mutator.mutator import Mutator


class StandardMutator(Mutator):

    """Standard mutator class."""

    def mutate(self, individual):
        """Mutates the individual."""
        actions = individual.copy().get_actions()

        if len(actions) == 0:
            return individual

        action_index = self.randomness.next_int(0, len(actions) - 1)
        action = actions[action_index].copy()
        action = self.apply_gaussian_mutation(action)
        individual.actions[action_index] = action
        return individual

    def apply_gaussian_mutation(self, action):
        """Applies the Gaussian mutation to the action."""
        sigma = self.config.get("mutation_sigma")
        pixels = action.get_color()
        delta = self.randomness.random_gaussian(pixels * 0, sigma)
        mutated_pixels = pixels + delta
        mutated_pixels = self.check_limit_values(mutated_pixels)
        action.set_color(mutated_pixels)
        return action
