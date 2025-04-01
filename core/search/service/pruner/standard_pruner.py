"""Standard pruner that tries to prune actions by removing each action at a time."""

from core.search.service.pruner.pruner import Pruner
from core.search.solution import Solution


class StandardPruner(Pruner):

    """Standard pruner that try to prune actions removing each actions at a time."""

    def pruner_type(self):
        """Return the type of the pruner."""
        return "standard"

    def minimize_actions_in_archive(self) -> Solution:
        """Minimize the actions in the archive."""

        self.ssu.start_minimization()
        solution = self.archive.extract_solution()

        actions = solution.actions.copy()
        actions_length = len(actions)

        total_length = actions_length
        remove_counter = 1

        counter = 0

        current_actions = actions.copy()

        while counter < actions_length:
            removed_action = current_actions.pop(counter)
            result = self.ff.calculate_fitness_with_actions(current_actions)

            if result.value == 0:
                self.ssu.remove_action(remove_counter, total_length, removed_action)
                actions_length -= 1
                counter -= 1
            else:
                self.ssu.kept_action(remove_counter, total_length, removed_action)
                current_actions.insert(counter, removed_action)

            counter += 1
            remove_counter += 1

        current_fitness = self.ff.calculate_fitness_with_actions(current_actions)
        return Solution(current_actions, current_fitness)
