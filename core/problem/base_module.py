"""Base module for the problem module."""
import logging

from dependency_injector import containers, providers

from core.config_parser import ConfigParser
from core.remote.remote_controller import RemoteController
from core.search.algorithms.search_algorithm import SearchAlgorithm
from core.search.service.adaptive_parameter_control import AdaptiveParameterControl
from core.search.service.archive import Archive
from core.search.service.fitness_function import FitnessFunction
from core.search.service.monitor.search_status_updater import SearchStatusUpdater
from core.search.service.monitor.statistics import Statistics
from core.search.service.mutator.mutator import Mutator
from core.search.service.randomness import Randomness
from core.search.service.sampler.sampler import Sampler
from core.search.service.search_time_controller import SearchTimeController


def configure_logger():
    """Configure the logger."""
    logger = logging.getLogger("optiattack")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


class BaseModule(containers.DeclarativeContainer):

    """Base module for the problem module."""

    config = providers.Configuration()
    config_parser = providers.Singleton(ConfigParser)
    randomness = providers.Singleton(Randomness, config=config)
    logger = providers.Singleton(configure_logger)
    stc = providers.Singleton(SearchTimeController, config=config)
    apc = providers.Singleton(AdaptiveParameterControl, stc=stc, config=config)
    remote_controller = providers.Singleton(RemoteController,
                                            config=config,
                                            stc=stc)
    archive = providers.Singleton(Archive,
                                  stc=stc,
                                  randomness=randomness,
                                  config=config)
    search_status_updater = providers.Singleton(SearchStatusUpdater,
                                                stc=stc,
                                                config=config,
                                                archive=archive)
    ff = providers.Singleton(FitnessFunction, archive=archive, remote_controller=remote_controller, stc=stc)
    mutator = providers.Singleton(Mutator, randomness=randomness, stc=stc, config=config, apc=apc)
    statistics = providers.Singleton(Statistics, stc=stc, archive=archive, config=config)
    sampler = providers.Singleton(Sampler,
                                  randomness=randomness,
                                  config=config)
    algorithm = providers.Singleton(SearchAlgorithm,
                                    ff=ff, randomness=randomness,
                                    stc=stc, archive=archive,
                                    config=config,
                                    mutator=mutator,
                                    sampler=sampler,
                                    apc=apc)
