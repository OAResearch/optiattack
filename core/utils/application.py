"""Configure the container with the provided configuration."""

from dependency_injector import providers

from core.config_parser import ConfigParser
from core.search.algorithms.mio_algorithm import MioAlgorithm
from core.search.algorithms.random_algorithm import RandomAlgorithm
from core.search.service.mutator.one_zero_mutator import OneZeroMutator
from core.search.service.mutator.standard_mutator import StandardMutator
from core.search.service.pruner.standard_pruner import StandardPruner
from core.search.service.sampler.random_sampler import RandomSampler
#from core.search.service.fitness_function.untargeted_fitness_function import TargetedFitnessFunction
from core.search.service.fitness_function.untargeted_fitness_function import UntargetedFitnessFunction


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

    if container.config.get("sampler") == ConfigParser.SamplerType.RANDOM_SAMPLER:
        container.sampler.override(providers.Singleton(RandomSampler,
                                                       randomness=container.randomness,
                                                       config=container.config
                                                       ))
    else:
        raise ValueError(f"Sampler {container.config.get('sampler')} not supported")

    # Fitness Function seçimi
    if container.config.get("target") != "None":
        container.ff.override(providers.Singleton(UntargetedFitnessFunction,
                                                  archive=container.archive,
                                                  remote_controller=container.remote_controller,
                                                  stc=container.stc,
                                                  target_label=container.config.get("target")))
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
    else:
        raise ValueError(f"Algorithm {current_algorithm} not supported")

    container.algorithm.override(providers.Singleton(algorithm,
                                                     ff=container.ff,
                                                     randomness=container.randomness,
                                                     stc=container.stc,
                                                     archive=container.archive,
                                                     config=container.config,
                                                     mutator=container.mutator,
                                                     sampler=container.sampler,
                                                     apc=container.apc))

    return container
