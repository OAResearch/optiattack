"""starting point of the application."""

from logging import Logger

from dependency_injector.wiring import inject, Provide

from core.problem.base_module import BaseModule
from core.remote.remote_controller import RemoteController
from core.search.algorithms.search_algorithm import SearchAlgorithm
from core.search.phase_controller import PhaseController
from core.search.service.adaptive_parameter_control import AdaptiveParameterControl
from core.search.service.archive import Archive
from core.search.service.fitness_function.fitness_function import FitnessFunction
from core.search.service.monitor.search_status_updater import SearchStatusUpdater
from core.search.service.monitor.statistics import Statistics
from core.search.service.mutator.mutator import Mutator
from core.search.service.pruner.pruner import Pruner
from core.search.service.randomness import Randomness
from core.search.service.search_time_controller import SearchTimeController
from core.utils.application import configure_container
from core.utils.images import read_image, resize_image, img_to_array, ProcessedImage


class OptiAttack:
    """Main class of the application."""

    @inject
    def __init__(self,
                 randomness: Randomness = Provide[BaseModule.randomness],
                 config: dict = Provide[BaseModule.config],
                 remote_controller:
                 RemoteController = Provide[BaseModule.remote_controller],
                 logger: Logger = Provide[BaseModule.logger],
                 stc: SearchTimeController = Provide[BaseModule.stc],
                 archive: Archive = Provide[BaseModule.archive],
                 search_status_updater: SearchStatusUpdater = Provide[BaseModule.search_status_updater],
                 ff: FitnessFunction = Provide[BaseModule.ff],
                 mutator: Mutator = Provide[BaseModule.mutator],
                 statistics: Statistics = Provide[BaseModule.statistics],
                 algorithm: SearchAlgorithm = Provide[BaseModule.algorithm],
                 apc: AdaptiveParameterControl = Provide[BaseModule.apc],
                 pruner: Pruner = Provide[BaseModule.pruner],
                 pc: PhaseController = Provide[BaseModule.pc]
                 ):
        """Initialize the application."""
        self.__name__ = "OptiAttack"

        self.randomness = randomness
        self.config = config
        self.remote_controller = remote_controller
        self.logger = logger
        self.stc = stc
        self.archive = archive
        self.search_status_updater = search_status_updater
        self.ff = ff
        self.mutator = mutator
        self.statistics = statistics
        self.algorithm = algorithm
        self.apc = apc
        self.pruner = pruner
        self.pc = pc

    def startup(self):
        """Prepare the application."""
        self.logger.info("Starting application")

        try:
            original_image = read_image(self.config.get("input_image"))
            resized_image = resize_image(original_image,
                                         self.config.get("image_width"),
                                         self.config.get("image_height"))
            image_as_array = img_to_array(resized_image)
            self.archive.set_image(ProcessedImage(original_image, resized_image, image_as_array))
        except FileNotFoundError:
            error_message = (f"Image file not found: "
                             f"{self.config.get('input_image')}")
            self.logger.error(error_message)
            quit()
        except Exception as e:
            error_message = f"Error reading image file: {e}"
            self.logger.error(error_message)
            quit()

        self.logger.info("Image loaded successfully")

        self.logger.info("Sending image to remote controller for testing")
        try:
            nut_info = self.remote_controller.run_nut(self.archive.image.array)
            if not nut_info["predictions"]:
                error_message = ("Error sending image to remote controller. "
                                 "Please check the remote controller or network under test.")
                self.logger.error(error_message)
                quit()
            self.archive.set_original_prediction_results(nut_info)
            self.logger.info(f"Connected to NUT: http://{nut_info['controller_host']}:{nut_info['controller_port']}")
        except ConnectionError:
            error_message = ("Connection error with remote controller. "
                             "Please check the remote controller or network under test.")
            self.logger.error(error_message)
            quit()
        except Exception as e:
            error_message = (f"Error sending image to remote controller: {e}. "
                             f"Please check the remote controller or network under test.")
            self.logger.error(error_message)
            quit()

        self.logger.info("Image sent to remote controller successfully. Ready to run application")

    def run(self):
        """Run the application."""

        self.pc.start()
        self.stc.start_search()
        self.algorithm.search()

        if self.config.get("enable_pruning") is True:
            self.pc.prune()
            final_solution = self.pruner.minimize_actions_in_archive()
            self.archive.set_minimized_solution(final_solution)
            self.stc.set_pruned_fitness(final_solution.fitness_value)

        self.pc.end()
        self.search_status_updater.search_end()
        self.statistics.write_statistics()
        return self.statistics.directories


if __name__ == "__main__":

    container = BaseModule()
    config_parser = container.config_parser()
    parsed_args = config_parser.parse_args()
    container.config.override(parsed_args)
    container = configure_container(container)

    if parsed_args.get("enable_ui"):
        from gradio_ui import WebApp
        webapp = WebApp(config_parser)
        webapp.web_app.launch(pwa=True, server_port=13000, server_name="0.0.0.0")
    else:
        app = OptiAttack()
        container.wire(modules=[app])
        app.startup()
        app.run()
