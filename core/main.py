"""starting point of the application."""

from logging import Logger

from dependency_injector.wiring import inject, Provide

from core.problem.base_module import BaseModule
from core.remote.remote_controller import RemoteController
from core.search.evaluated_individual import EvaluatedIndividual
from core.search.fitness_value import FitnessValue
from core.search.service.archive import Archive
from core.search.service.monitor.search_status_updater import SearchStatusUpdater
from core.search.service.randomness import Randomness
from core.search.service.search_time_controller import SearchTimeController
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
                 ff=Provide[BaseModule.ff],
                 mutator=Provide[BaseModule.mutator]
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
        self.logger.info("Running application")

        self.stc.start_search()

        while self.stc.should_continue_search():
            individual = self.archive.sample_individual()

            fitness_value = SearchTimeController.measure_time_millis(
                self.log_execution_time,
                lambda: self.ff.evaluate(individual)
            )
            ei = EvaluatedIndividual(individual, fitness_value)
            self.archive.add_archive_if_needed(ei)

    def log_execution_time(self, t: int, ind: FitnessValue):
        """Log the execution time and update the individual's execution time."""
        self.stc.report_executed_individual_time(t, 1)
        ind.set_execution_time_ms(t)


if __name__ == "__main__":

    container = BaseModule()
    config_parser = container.config_parser()
    parsed_args = config_parser.parse_args()
    container.config.override(parsed_args)

    app = OptiAttack()
    container.wire(modules=[app])
    app.startup()
    app.run()
