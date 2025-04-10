"""Abstract class for search algorithms."""
from core.search.service.adaptive_parameter_control import AdaptiveParameterControl
from core.search.service.archive import Archive
from core.search.service.fitness_function.fitness_function import FitnessFunction
from core.search.service.mutator.mutator import Mutator
from core.search.service.randomness import Randomness
from core.search.service.sampler.sampler import Sampler
from core.search.service.search_time_controller import SearchTimeController


class SearchAlgorithm:

    """Abstract class for search algorithms."""

    def __init__(self, ff: FitnessFunction,
                 randomness: Randomness,
                 stc: SearchTimeController,
                 archive: Archive,
                 config: dict,
                 mutator: Mutator,
                 sampler: Sampler,
                 apc: AdaptiveParameterControl):
        """Initialize the search algorithm."""

        self.ff = ff
        self.randomness = randomness
        self.stc = stc
        self.archive = archive
        self.config = config
        self.mutator = mutator
        self.sampler = sampler
        self.apc = apc

    def get_type(self):
        """Return the type of the search algorithm."""
        pass

    def search_once(self):
        """Search for a solution."""
        pass

    def setup_before_search(self):
        """Setup before the search."""
        pass

    def search(self):
        """All the search process."""

        self.stc.start_search()
        self.setup_before_search()

        while self.stc.should_continue_search():
            self.search_once()

        solution = self.after_search()
        return solution

    def after_search(self):
        """Actions to do after the search."""
        return self.archive.extract_solution()
