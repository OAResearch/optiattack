"""starting point of the application."""

from logging import Logger
from dependency_injector.wiring import inject, Provide

from core.problem.base_module import BaseModule
from core.remote.remote_controller import RemoteController
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
                 search_time_controller: SearchTimeController = Provide[BaseModule.search_time_controller]
                 ):
        """Initialize the application."""
        self.__name__ = "OptiAttack"

        self.randomness = randomness
        self.config = config
        self.remote_controller = remote_controller
        self.logger = logger
        self.search_time_controller = search_time_controller

        self.image = None

    def startup(self):
        """Prepare the application."""
        self.logger.info("Starting application")

        try:
            original_image = read_image(self.config.get("input_image"))
            resized_image = resize_image(original_image,
                                         self.config.get("image_width"),
                                         self.config.get("image_height"))
            image_as_array = img_to_array(resized_image)
            self.image = ProcessedImage(original_image, resized_image, image_as_array)
        except FileNotFoundError:
            error_message = (f"Image file not found: "
                             f"{self.config.get('input_image')}")
            self.logger.error(error_message)
            raise FileNotFoundError(f"Image file not found: {self.config.get('input_image')}")
        except Exception as e:
            error_message = f"Error reading image file: {e}"
            self.logger.error(error_message)
            raise Exception(error_message)

        self.logger.info("Image loaded successfully")

        self.logger.info("Sending image to remote controller for testing")
        try:
            self.remote_controller.new_action(self.image.array)
        except Exception as e:
            error_message = (f"Error sending image to remote controller: {e}. "
                             f"Please check the remote controller or network under test.")
            self.logger.error(error_message)
            raise Exception(error_message)

        self.logger.info("Image sent to remote controller successfully. Ready to run application")

    def run(self):
        """Run the application."""
        self.logger.info("Running application")

        for i in range(100):
            self.remote_controller.new_action(self.image.array)
            self.logger.info(f"Action {i} sent to remote controller")


if __name__ == "__main__":

    container = BaseModule()
    config_parser = container.config_parser()
    parsed_args = config_parser.parse_args()
    container.config.override(parsed_args)

    app = OptiAttack()
    container.wire(modules=[app])
    app.startup()
    app.run()
