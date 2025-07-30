"""Configure the container with the provided configuration."""

from dependency_injector import providers

from core.config_parser import ConfigParser
from core.search.algorithms.genetic_algorithm import GeneticAlgorithm
from core.search.algorithms.mio_algorithm import MioAlgorithm
from core.search.algorithms.random_algorithm import RandomAlgorithm
from core.search.service.crossover.single_point_crossover import SinglePointCrossover
from core.search.service.mutator.one_zero_mutator import OneZeroMutator
from core.search.service.mutator.standard_mutator import StandardMutator
from core.search.service.pruner.standard_pruner import StandardPruner
from core.search.service.sampler.gaussian_sampler import GaussianSampler
from core.search.service.sampler.random_sampler import RandomSampler
from core.search.service.fitness_function.untargeted_fitness_function import UntargetedFitnessFunction
from core.search.service.fitness_function.targeted_fitness_function import TargetedFitnessFunction


def configure_container(container):
    """Configure the container with the provided configuration."""

    if container.config.get("mutator") == ConfigParser.Mutators.STANDARD_MUTATOR:
        container.mutator.override(providers.Singleton(StandardMutator,
                                                       randomness=container.randomness,
                                                       stc=container.stc,
                                                       config=container.config,
                                                       apc=container.apc))
    elif container.config.get("mutator") == ConfigParser.Mutators.ONE_ZERO_MUTATOR:
        container.mutator.override(providers.Singleton(OneZeroMutator,
                                                       randomness=container.randomness,
                                                       stc=container.stc,
                                                       config=container.config,
                                                       apc=container.apc))
    else:
        raise ValueError(f"Mutator {container.config.get('mutator')} not supported")

    if container.config.get("crossover") == ConfigParser.Crossovers.SINGLE_POINT_CROSSOVER:
        container.crossover.override(providers.Singleton(SinglePointCrossover,
                                                         randomness=container.randomness,
                                                         stc=container.stc,
                                                         config=container.config,
                                                         apc=container.apc))
    else:
        raise ValueError(f"Crossover {container.config.get('crossover')} not supported")

    if container.config.get("sampler") == ConfigParser.SamplerType.RANDOM_SAMPLER:
        container.sampler.override(providers.Singleton(RandomSampler,
                                                       randomness=container.randomness,
                                                       config=container.config,
                                                       archive=container.archive
                                                       ))
    elif container.config.get("sampler") == ConfigParser.SamplerType.GAUSSIAN_SAMPLER:

        container.sampler.override(providers.Singleton(GaussianSampler,
                                                       randomness=container.randomness,
                                                       config=container.config,
                                                       archive=container.archive
                                                       ))
    else:
        raise ValueError(f"Sampler {container.config.get('sampler')} not supported")

    if container.config.get("attack_type") == ConfigParser.AttackType.TARGETED:
        container.ff.override(providers.Singleton(TargetedFitnessFunction,
                                                  archive=container.archive,
                                                  remote_controller=container.remote_controller,
                                                  stc=container.stc,
                                                  target=container.config.get("target")
                                                  ))
    else:
        container.ff.override(providers.Singleton(UntargetedFitnessFunction,
                                                  archive=container.archive,
                                                  remote_controller=container.remote_controller,
                                                  stc=container.stc))

    if container.config.get("pruning_method") == ConfigParser.PruningTypes.STANDARD:
        container.pruner.override(providers.Singleton(StandardPruner,
                                                      archive=container.archive,
                                                      ff=container.ff,
                                                      ssu=container.search_status_updater))
    else:
        raise ValueError(f"Pruning method {container.config.get('pruning_method')} not supported")

    current_algorithm = container.config.get("algorithm")

    # TODO switch/case can be used but it is supported after python 3.10

    if current_algorithm == ConfigParser.Algorithms.RANDOM_SEARCH:
        algorithm = RandomAlgorithm
    elif current_algorithm == ConfigParser.Algorithms.MIO:
        algorithm = MioAlgorithm
    elif current_algorithm == ConfigParser.Algorithms.GENETIC:
        algorithm = GeneticAlgorithm
    else:
        raise ValueError(f"Algorithm {current_algorithm} not supported")

    container.algorithm.override(providers.Singleton(algorithm,
                                                     ff=container.ff,
                                                     randomness=container.randomness,
                                                     stc=container.stc,
                                                     archive=container.archive,
                                                     config=container.config,
                                                     mutator=container.mutator,
                                                     crossover=container.crossover,
                                                     sampler=container.sampler,
                                                     apc=container.apc))

    return container
