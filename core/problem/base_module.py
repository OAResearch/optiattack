"""Base module for the problem module."""
import logging

from dependency_injector import containers, providers

from core.config_parser import ConfigParser
from core.remote.remote_controller import RemoteController
from core.search.service.archive import Archive
from core.search.service.fitness_function import FitnessFunction
from core.search.service.monitor.search_status_updater import SearchStatusUpdater
from core.search.service.monitor.statistics import Statistics
from core.search.service.mutator.standard_mutator import StandardMutator
from core.search.service.randomness import Randomness
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
    mutator = providers.Singleton(StandardMutator, randomness=randomness, stc=stc, config=config)
    statistics = providers.Singleton(Statistics, stc=stc, archive=archive, config=config)
