"""Differential search algorithm implementation."""

from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.algorithm import Algorithm
from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.evaluator import Evaluator
from pymoo.core.initialization import Initialization
from pymoo.core.mating import Mating
from pymoo.core.mutation import Mutation
from pymoo.core.population import Population
from pymoo.core.repair import NoRepair
from pymoo.core.sampling import Sampling
from pymoo.core.selection import Selection
from pymoo.util.optimum import filter_optimum

from core.search.algorithms.search_algorithm import SearchAlgorithm
from core.search.algorithms.utils.pixel_change_minimise import PixelChangeMinimise


class PyMooAlgorithm(SearchAlgorithm):

    """PyMoo algorithms implementation."""

    def __init__(self,
                 selection: Selection,
                 crossover: Crossover,
                 mutation: Mutation,
                 sampling: Sampling,
                 eliminate_duplicates: ElementwiseDuplicateElimination,
                 *args, **kwargs):
        """Initialize PyMoo algorithm."""
        super().__init__(*args, **kwargs)

        self.algorithm = None
        self.population_size = 0
        self.n_offsprings = 0
        self.pop: Population
        self.n_iter = 1
        self.is_initialized = False
        self.off = None
        self.survival = FitnessSurvival()
        self.problem = PixelChangeMinimise(config=self.config)
        self.repair = NoRepair()

        self.mating = Mating(selection,
                             crossover,
                             mutation,
                             repair=self.repair,
                             eliminate_duplicates=eliminate_duplicates,
                             n_max_iterations=100)

        self.evaluator = Evaluator()
        self.initialization = Initialization(sampling,
                                             repair=self.repair,
                                             eliminate_duplicates=eliminate_duplicates)

    def setup_after_init(self, algorithm: Algorithm, population_size: int):
        """Set the algorithm to be used."""
        self.algorithm = algorithm
        self.population_size = population_size
        self.n_offsprings = population_size

    def setup_before_search(self):
        """Setup before starting the search."""
        self.problem.set_image_as_array(self.archive.get_image())
        self.problem.set_fitness_function(self.ff)
        self.problem.set_archive(self.archive)

    def search_once(self):
        """Execute one iteration of the search."""
        # get the infill solutions
        infills = self.infill()

        # call the advance with them after evaluation
        if infills is not None:
            self.evaluator.eval(self.problem, infills, algorithm=self)
            self.advance(infills=infills)

        # if the algorithm does not follow the infill-advance scheme just call advance
        else:
            self.advance()

    def advance(self, infills=None, **kwargs):
        """Advance the algorithm with the provided infills or continue the search."""

        # if infills have been provided set them as offsprings and feed them into advance
        self.off = infills

        # if the algorithm has not been already initialized
        if not self.is_initialized:
            self.n_iter = 1
            self.pop = infills
            self.is_initialized = True
            self._set_optimum()
            self.n_iter += 1

        else:

            # call the implementation of the advance method - if the infill is not None
            val = self._advance(infills=infills, **kwargs)

            # always advance to the next iteration - except if the algorithm returns False
            if val is None or val:
                self._set_optimum()
                self.n_iter += 1
        ret = self.opt
        return ret

    def _advance(self, infills=None, **kwargs):
        """Advance the algorithm with the provided infills or continue the search."""

        # the current population
        pop = self.pop

        # merge the offsprings with the current population
        if infills is not None:
            pop = Population.merge(self.pop, infills)

        # execute the survival to find the fittest solutions
        self.pop = self.survival._do(None, pop, n_survive=self.population_size)

    def _set_optimum(self):
        """Set the optimum solution from the current population."""

        self.opt = filter_optimum(self.pop, least_infeasible=True)

    def infill(self):
        """Generate new solutions for the next iteration of the algorithm."""
        # the first time next is called simply initial the algorithm - makes the interface cleaner
        if not self.is_initialized:

            # hook mostly used by the class to happen before even to initialize
            self._initialize()

            # execute the initialization infill of the algorithm
            infills = self._initialize_infill()

        else:
            # request the infill solutions if the algorithm has implemented it
            infills = self._infill()

        # set the current generation to the offsprings
        if infills is not None:
            infills.set("n_gen", self.n_iter)
            infills.set("n_iter", self.n_iter)

        return infills

    def _infill(self):
        """Generate new solutions for the next iteration of the algorithm."""
        # do the mating using the current population
        off = self.mating.do(self.problem, self.pop, self.n_offsprings)

        return off

    def _initialize_infill(self):
        """Initialize the population for the first iteration of the algorithm."""
        # return self.init_population()
        pop = self.initialization.do(self.problem, self.population_size, algorithm=self.algorithm)
        return pop

    def _initialize(self):
        """Initialize the algorithm before the first iteration."""
        # set the attribute for the optimization method to start
        self.n_iter = 1
        self.pop = Population.empty()
        self.opt = None
