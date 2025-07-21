"""MOSA search algorithm implementation."""
import math
from typing import List, Set, Dict, Optional
from dataclasses import dataclass

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival, comp_by_cv_and_fitness, GA
from pymoo.core.evaluator import Evaluator
from pymoo.core.initialization import Initialization
from pymoo.core.mating import Mating
from pymoo.core.population import Population
from pymoo.core.repair import NoRepair
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.util.optimum import filter_optimum

from core.config_parser import ConfigParser
from core.search.algorithms.search_algorithm import SearchAlgorithm
from core.search.algorithms.utils.crossovers import ArraySlice
from core.search.algorithms.utils.de_mutations import PixelMutation
from core.search.algorithms.utils.eliminations import SimpleElimination
from core.search.algorithms.utils.pixel_change_minimise import PixelChangeMinimise
from core.search.algorithms.utils.sampling import InitialImageSampling
from core.search.evaluated_individual import EvaluatedIndividual


# @dataclass
# class IndividualData:
#     """Data class to store individual information."""
#     individual: any
#     rank: int = -1
#     crowding_distance: float = -1.0


class GADEAlgorithm(SearchAlgorithm):
    """Differential evolution algorithm implementation."""

    def __init__(self, *args, **kwargs):
        """Initialize SGA algorithm."""
        super().__init__(*args, **kwargs)
        self.population_size = self.config.get("population_size")
        self.population: Population
        self.n_iter = 1
        self.is_initialized = False
        self.off = None
        self.survival = FitnessSurvival()
        self.selection = TournamentSelection(func_comp=comp_by_cv_and_fitness)
        self.crossover = ArraySlice()
        self.mutation = PixelMutation()
        self.eliminate_duplicates = SimpleElimination()
        self.problem = PixelChangeMinimise(config=self.config)
        self.sampling = InitialImageSampling()
        self.repair = NoRepair()
        self.mating = Mating(self.selection,
                        self.crossover,
                        self.mutation,
                        repair=self.repair,
                        eliminate_duplicates=self.eliminate_duplicates,
                        n_max_iterations=100)
        self.n_offsprings = self.population_size

        self.algorithm = GA(pop_size=self.population_size,
                   sampling=self.sampling,
                   crossover=ArraySlice(),
                   mutation=PixelMutation(),
                   eliminate_duplicates=SimpleElimination())
        self.evaluator = Evaluator()
        self.initialization = Initialization(self.sampling,
                                             repair=self.repair,
                                             eliminate_duplicates=self.eliminate_duplicates)



    def get_type(self):
        """Return the type of the search algorithm."""
        return ConfigParser.Algorithms.GA_DE

    def setup_before_search(self):
        """Setup before starting the search."""
        self.problem.set_image_as_array(self.archive.get_image())
        self.problem.set_fitness_function(self.ff)
        self.problem.set_archive(self.archive)

    def init_population(self):
        """Initialize the population with random individuals."""
        self.n_iter = 1
        population = np.empty(self.population_size, dtype=EvaluatedIndividual)

        for _ in range(self.population_size):
            if not self.stc.should_continue_search():
                break
            individual = self.sampler.sample()
            evaluated = self.ff.calculate_fitness(individual)
            if evaluated:
                self.archive.add_archive_if_needed(evaluated)
                population[_] = evaluated
        pop = Population.empty(self.population_size)
        pop.set("X", population)
        f_values = np.array([ind.fitness.value for ind in population])
        pop.set("F", f_values[:, None])  # Assuming single objective optimization
        return pop

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

        # the current population
        pop = self.pop

        # merge the offsprings with the current population
        if infills is not None:
            pop = Population.merge(self.pop, infills)

        # execute the survival to find the fittest solutions
        self.pop = self.survival._do(None, pop, n_survive=self.population_size)

    def _set_optimum(self):
        self.opt = filter_optimum(self.pop, least_infeasible=True)

    def infill(self):

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

        # do the mating using the current population
        off = self.mating.do(self.problem, self.pop, self.n_offsprings)

        # if the mating could not generate any new offspring (duplicate elimination might make that happen)
        if len(off) == 0:
            self.termination.force_termination = True
            return

        # if not the desired number of offspring could be created
        elif len(off) < self.n_offsprings:
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")

        return off

    def mating(self, pop, n_offsprings):
        # the population object to be used
        off = Population.create()

        # infill counter - counts how often the mating needs to be done to fill up n_offsprings
        n_infills = 0
        # iterate until enough offsprings are created
        while len(off) < n_offsprings:

            # how many offsprings are remaining to be created
            n_remaining = n_offsprings - len(off)

            # do the mating
            _off = self.handle_mating(pop, n_remaining)

            # eliminate the duplicates
            _off = self.eliminate_duplicates.do(_off, pop, off)

            # if more offsprings than necessary - truncate them randomly
            if len(off) + len(_off) > n_offsprings:

                # IMPORTANT: Interestingly, this makes a difference in performance for some algorithms
                n_remaining = n_offsprings - len(off)
                _off = _off[:n_remaining]

            # add to the offsprings and increase the mating counter
            off = Population.merge(off, _off)
            n_infills += 1

        return off

    def handle_mating(self, pop, n_offsprings, parents=None):

        # how many parents need to be select for the mating - depending on number of offsprings remaining
        n_matings = math.ceil(n_offsprings / self.crossover.n_offsprings)

        # if the parents for the mating are not provided directly - usually selection will be used
        if parents is None:

            # select the parents for the mating - just an index array
            parents = self.selection(pop, n_matings, n_parents=self.crossover.n_parents)

        # do the crossover using the parents index and the population - additional data provided if necessary
        off = self.crossover(parents)

        # do the mutation on the offsprings created through crossover
        off = self.mutation(off)

        return off

    def _initialize_infill(self):
        # return self.init_population()
        pop = self.initialization.do(self.problem, self.population_size, algorithm=self.algorithm)
        return pop

    def _initialize(self):

        # set the attribute for the optimization method to start
        self.n_iter = 1
        self.pop = Population.empty()
        self.opt = None

    def selection(self):
        """Perform tournament selection."""
        pressure = 2
        n_parents = 2
        n_select = int(self.population_size / n_parents)

        n_random = n_select * n_parents * pressure

        # number of permutations needed
        n_perms = math.ceil(n_random / self.population_size)

        P = []
        for i in range(n_perms):
            P.append(np.random.permutation(self.population_size))
        P = np.concatenate(P)

        P = P[:self.population_size * 2]
        P = np.reshape(P, (self.population_size, 2))
        S = self.comp_by_cv_and_fitness(self.population, P)
        selected = np.reshape(S, (n_select, n_parents))

        return self.population[selected]

        # min_idx = self.randomness.next_int(0, len(self.population) - 1)
        #
        # for _ in range(self.config.get("tournament_size") - 1):
        #     sel = self.randomness.next_int(0, len(self.population) - 1)
        #     if self.population[sel].rank < self.population[min_idx].rank:
        #         min_idx = sel
        #     elif (self.population[sel].rank == self.population[min_idx].rank and
        #           self.population[sel].crowding_distance < self.population[min_idx].crowding_distance):
        #         min_idx = sel
        #
        # return self.population[min_idx]

    # pymoo\algorithms\soo\nonconvex\ga.py
    def comp_by_cv_and_fitness(self, pop, P, **kwargs):
        S = np.full(P.shape[0], np.nan)

        for i in range(P.shape[0]):
            a, b = P[i, 0], P[i, 1]
            # if at least one solution is infeasible
            # if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            #     S[i] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True)
            #
            # # both solutions are feasible just set random
            # else:
            #     S[i] = compare(a, pop[a].F, b, pop[b].F, method='smaller_is_better', return_random_if_equal=True)

            S[i] = self.compare(a, pop[a].fitness.value, b, pop[b].fitness.value, method='larger_is_better', return_random_if_equal=True)

        return S[:, None].astype(int)

    # pymoo\operators\selection\tournament.py
    def compare(self, a, a_val, b, b_val, method, return_random_if_equal=False):
        if method == 'larger_is_better':
            if a_val > b_val:
                return a
            elif a_val < b_val:
                return b
            else:
                if return_random_if_equal:
                    return np.random.choice([a, b])
                else:
                    return None
        elif method == 'smaller_is_better':
            if a_val < b_val:
                return a
            elif a_val > b_val:
                return b
            else:
                if return_random_if_equal:
                    return np.random.choice([a, b])
                else:
                    return None
        else:
            raise Exception("Unknown method.")
