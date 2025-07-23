"""AGEMOEA search algorithm implementation."""

from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.soo.nonconvex.ga import comp_by_cv_and_fitness
from pymoo.core.population import Population
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.util.optimum import filter_optimum

from core.config_parser import ConfigParser
from core.search.algorithms.pymoo_algorithm import PyMooAlgorithm
from core.search.algorithms.utils.crossovers import ArraySlice
from core.search.algorithms.utils.de_mutations import PixelMutation
from core.search.algorithms.utils.eliminations import SimpleElimination
from core.search.algorithms.utils.sampling import InitialImageSampling


class AGEMOEAAlgorithm(PyMooAlgorithm):

    """AGEMOEA algorithm implementation."""

    def __init__(self, *args, **kwargs):
        """Initialize SGA algorithm."""

        selection = TournamentSelection(func_comp=comp_by_cv_and_fitness)
        crossover = ArraySlice()
        mutation = PixelMutation()
        eliminate_duplicates = SimpleElimination()
        sampling = InitialImageSampling()

        super().__init__(selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         sampling=sampling,
                         eliminate_duplicates=eliminate_duplicates,
                         *args, **kwargs)
        population_size = self.config.get("population_size")

        algorithm = AGEMOEA(pop_size=population_size,
                            sampling=sampling,
                            crossover=crossover,
                            mutation=mutation,
                            eliminate_duplicates=eliminate_duplicates)

        self.setup_after_init(algorithm=algorithm, population_size=population_size)

    def get_type(self):
        """Return the type of the search algorithm."""
        return ConfigParser.Algorithms.AGEMOEA

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
        """Advance the algorithm with the provided infills."""

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
        """Advance the algorithm with the provided infills."""
        # the current population
        pop = self.pop

        # merge the offsprings with the current population
        if infills is not None:
            pop = Population.merge(self.pop, infills)

        # execute the survival to find the fittest solutions
        self.pop = self.survival._do(None, pop, n_survive=self.population_size)

    def _set_optimum(self):
        """Set the optimum of the algorithm based on the current population."""
        self.opt = filter_optimum(self.pop, least_infeasible=True)

    def infill(self):
        """Generate infill solutions for the algorithm."""
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
        """Generate infill solutions for the algorithm."""
        # do the mating using the current population
        off = self.mating.do(self.problem, self.pop, self.n_offsprings)
        return off

    def _initialize_infill(self):
        """Initialize the infill solutions for the algorithm."""
        # return self.init_population()
        pop = self.initialization.do(self.problem, self.population_size, algorithm=self.algorithm)
        return pop

    def _initialize(self):
        """Initialize the algorithm before the first search iteration."""
        # set the attribute for the optimization method to start
        self.n_iter = 1
        self.pop = Population.empty()
        self.opt = None
