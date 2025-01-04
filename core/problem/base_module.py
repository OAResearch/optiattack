"""Base module for the problem module."""
import logging

from dependency_injector import containers, providers

from core.config_parser import ConfigParser
from core.remote.remote_controller import RemoteController
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
    remote_controller = providers.Singleton(RemoteController, config=config)
    logger = providers.Singleton(configure_logger)
    search_time_controller = providers.Singleton(SearchTimeController, config=config)
