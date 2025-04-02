import os
import sys

import pytest

from client.optiattack_client import collect_info
from core.problem.base_module import BaseModule
from core.utils.application import configure_container
from main import OptiAttack

PROCESS_IMAGE_RESPONSE = {
    "predictions": [
        {"label": "zebra", "score": 0.99},
        {"label": "horse", "score": 0.01},
    ]
}

@collect_info(host="localhost", port=38010)
def process_image(encoded_image: bytes):
    return PROCESS_IMAGE_RESPONSE

@pytest.fixture
def container():
    # set arguments for the argparse
    sys.argv = [
        "main.py",
        # "--input_image",
        # "../../test_img.jpeg",
        "--show_plots",
        "false",
        "--nut_port",
        "38010"
    ]
    container = BaseModule()
    config_parser = container.config_parser()
    parsed_args = config_parser.parse_args()
    container.config.override(parsed_args)
    container = configure_container(container)
    yield container


    # container = BaseModule()
    # container.unwire()
    #
    # container.config.override(
    #     {"seed": 42, "nut_host": "localhost", "nut_port": 38010,
    #      "base_endpoint": "/api/v1", "mutator": ConfigParser.Mutators.STANDARD_MUTATOR, "sampler": ConfigParser.SamplerType.RANDOM_SAMPLER,
    #      "pruning_method": ConfigParser.PruningTypes.STANDARD, "algorithm": ConfigParser.Algorithms.MIO,
    #      "input_image": "../../test_img.jpeg", "image_width": 224, "image_height": 224,
    #      "stopping_criterion": ConfigParser.StoppingCriterion.INDIVIDUAL_EVALUATIONS,
    #      "max_evaluations": 1000, "min_action_size": 1, "max_action_size": 10, "snapshot_interval": 10,
    #      "random_sampling_probability": 0.5, "apc_start_time": 0.5, "apc_threshold": 0.5, "focused_search_activation_time": 0.5})
    #
    # # config_parser = container.config_parser()
    # # parsed_args = config_parser.parse_args()
    # # container.config.override(parsed_args)
    # container = configure_container(container)
    # app = OptiAttack()
    # container.wire(modules=[app])
    # app.startup()
    # app.run()
    # yield app

def test_run_nut(container):
    app = OptiAttack()
    container.wire(modules=[app])
    app.startup()
    app.run()

    output_folder = app.config.get("output_dir")
    experiments_folder = app.config.get("experiment_label")
    seed = app.config.get("seed")

    save_folder = f"{output_folder}/{experiments_folder}/{seed}"
    # is file exists assert
    assert os.path.exists(save_folder)
    assert os.path.exists(f"{save_folder}/images")
    assert os.path.exists(f"{save_folder}/statistics")

    # check if the image is saved
    assert os.path.exists(f"{save_folder}/images/final_image.jpg")
    assert os.path.exists(f"{save_folder}/images/line.png")
    assert os.path.exists(f"{save_folder}/images/matrix_overlay.png")

    # check if the statistics are saved
    assert os.path.exists(f"{save_folder}/statistics/data.json")



