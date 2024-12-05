"""starting point of the application."""

from io import BytesIO

import numpy as np
from PIL import Image
from dependency_injector.wiring import inject, Provide

from core.problem.base_module import BaseModule
from core.remote.remote_controller import RemoteController
from core.search.service.randomness import Randomness


class OptiAttack:

    """Main class of the application."""

    @inject
    def __init__(self,
                 randomness: Randomness = Provide[BaseModule.randomness],
                 config: dict = Provide[BaseModule.config],
                 remote_controller:
                 RemoteController = Provide[BaseModule.remote_controller]
                 ):
        """Initialize the application."""
        self.randomness = randomness
        self.config = config
        self.remote_controller = remote_controller
        self.__name__ = "OptiAttack"

    def run(self):
        """Run the application."""
        print("Running application")

        image_data = BytesIO()
        image = Image.new("RGB", (224, 224), color="red")
        image.save(image_data, format="JPEG")
        image_data.seek(0)

        image_array = np.array(image)

        for i in range(100):
            self.remote_controller.new_action(image_array)
            print(f"Action {i} completed")


if __name__ == "__main__":

    container = BaseModule()
    config_parser = container.config_parser()
    parsed_args = config_parser.parse_args()
    container.config.override(parsed_args)

    app = OptiAttack()
    container.wire(modules=[app])
    app.run()
