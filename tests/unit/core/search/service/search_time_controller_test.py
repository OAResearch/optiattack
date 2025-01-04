from time import sleep

import pytest

from core.config_parser import ConfigParser
from core.main import OptiAttack
from core.problem.base_module import BaseModule


@pytest.fixture
def container():
    container = BaseModule()
    yield container
    container.unwire()


def get_app(container, config_setting=None):
    container.config.override(config_setting)
    app = OptiAttack(
        config=container.config(),
        randomness=container.randomness(),
        search_time_controller=container.search_time_controller(),
    )
    assert app.config.get("stopping_criterion") == config_setting.get("stopping_criterion")
    assert app.config.get("max_evaluations") == config_setting.get("max_evaluations")
    return app


def test_individual_evaluations(container):
    app = get_app(container,
                  {"stopping_criterion": ConfigParser.StoppingCriterion.INDIVIDUAL_EVALUATIONS,
                   "max_evaluations": 100,
                   "seed": 42})

    app.search_time_controller.start_search()
    for i in range(100):
        app.search_time_controller.new_individual_evaluation()

    assert app.search_time_controller.should_continue_search() is False


def test_time(container):
    app = get_app(container,
                  {"stopping_criterion": ConfigParser.StoppingCriterion.TIME,
                   "max_evaluations": 10,
                   "seed": 42})

    app.search_time_controller.start_search()
    sleep(10)
    assert app.search_time_controller.percentage_used_budget() >= 1.0
    assert app.search_time_controller.should_continue_search() is False
