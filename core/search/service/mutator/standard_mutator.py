"""Standard mutator module."""
from core.search.action import Action
from core.search.service.mutator.mutator import Mutator


class StandardMutator(Mutator):

    """Standard mutator class."""

    def mutate(self, individual):
        """Mutates the individual."""
        mutated_individual = individual.copy()
        actions = mutated_individual.copy().get_actions()

        if len(actions) == 0:
            return mutated_individual

        action_index = self.randomness.next_int(0, len(actions))
        action = actions[action_index].copy()
        action = self.apply_gaussian_mutation(action)
        mutated_individual.actions[action_index] = action
        return mutated_individual

    def apply_gaussian_mutation(self, action: Action):
        """Applies the Gaussian mutation to the action."""

        sigma = self.apc.get_pixel_apc()
        pixels = action.get_color()
        delta = self.randomness.random_gaussian(pixels * 0, sigma)
        mutated_pixels = pixels + delta
        mutated_pixels = self.check_limit_values(mutated_pixels)
        action.set_color(mutated_pixels)

        sigma = self.apc.get_location_apc()
        mutated_width = self.randomness.random_gaussian(action.get_location()[0], sigma)
        mutated_height = self.randomness.random_gaussian(action.get_location()[1], sigma)
        mutated_width, mutated_height = self.check_location_limits(mutated_width, mutated_height)
        action.set_location(mutated_width, mutated_height)

        return action
