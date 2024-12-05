"""Base module for the problem module."""

from dependency_injector import containers, providers

from core.config_parser import ConfigParser
from core.remote.remote_controller import RemoteController
from core.search.service.randomness import Randomness


class BaseModule(containers.DeclarativeContainer):

    """Base module for the problem module."""

    config = providers.Configuration()
    config_parser = providers.Singleton(ConfigParser)
    randomness = providers.Singleton(Randomness, config=config)
    remote_controller = providers.Singleton(RemoteController, config=config)
