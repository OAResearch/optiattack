"""Configuration file for the Client."""
import argparse
import inspect
from typing import Any

from core.utils.decorators import cfg


class ConfigParser:

    """Configuration parser for the Client."""

    def __init__(self):
        """Initialize the configuration parser."""
        self.parser = argparse.ArgumentParser(
            description="Application Configuration")
        self._args = None
        self._defaults = {}
        self._descriptions = {}

        # Add parameters to the parser
        for name, obj in inspect.getmembers(self, inspect.ismethod):
            if hasattr(obj, "cfg_description"):
                self.add_param(name, obj(), obj.cfg_description)

    def add_param(self, name: str, default: Any, description: str, **kwargs):
        """Add a parameter to the argument parser."""
        # Save default values and descriptions
        self._defaults[name] = default
        self._descriptions[name] = description

        # Add argument to the parser
        self.parser.add_argument(f"--{name}",
                                 default=default,
                                 help=description,
                                 type=type(default),
                                 **kwargs)

    def parse_args(self):
        """Parse command-line arguments and validate them."""
        args = self.parser.parse_args()
        self.validate_args(args)
        return vars(args)

    # def parse_args(self, args):
    #     """
    #     Parse command-line arguments and update defaults.
    #     """
    #     if self._args is None:
    #         # Parse arguments
    #         self._args = vars(self.parser.parse_args(args))
    #         # Update defaults with command-line arguments
    #         for key, value in self._args.items():
    #             self._defaults[key] = value
    #     return self._args

    def to_markdown(self, output_file: str = "config.md"):
        """Export configuration parameters to a Markdown file."""
        with open(output_file, "w") as f:
            f.write("# Configuration Parameters\n\n")
            for name, default in self._defaults.items():
                description = self._descriptions.get(
                    name, "No description provided.")
                f.write(f"## {name}\n\n")
                f.write(f"- **Default Value**: {default}\n")
                f.write(f"- **Description**: {description}\n\n")

    def default_params(self):
        """Return default parameters as a dictionary."""
        return self._defaults

    @staticmethod
    def validate_args(args):
        """Validate the arguments."""
        pass

    # Define parameters using Cfg decorator
    @cfg("Host address for the NUT. Default is 'localhost'.")
    def nut_host(self):
        """Host address for the NUT."""
        return "localhost"

    @cfg("Port number for the NUT. Default is 38000.")
    def nut_port(self):
        """Port number for the NUT."""
        return 38000

    @cfg("Base endpoint for the NUT. Default is '/api/v1'.")
    def base_endpoint(self):
        """Base endpoint for the NUT."""
        return "/api/v1"

    @cfg("Seed number for the random number generator. "
         "Negative values mean use the system time.")
    def seed(self):
        """Seed number for the random number generator."""
        return -1

    @cfg("Path to the input image.")
    def input_image(self):
        """Path to the input image."""
        return "../tests/test_img.jpeg"

    @cfg("Image width in pixels. Should be same as the model input size.")
    def image_width(self):
        """Image width in pixels."""
        return 224

    @cfg("Image height in pixels. Should be same as the model input size.")
    def image_height(self):
        """Image height in pixels."""
        return 224

    class StoppingCriterion:

        """Stopping criterion for the search."""

        INDIVIDUAL_EVALUATIONS = "individual_evaluations"
        TIME = "time"

    @cfg("Stopping criterion for the search. "
         "Options: 'individual_evaluations' or 'time'.")
    def stopping_criterion(self):
        """Stopping criterion for the search."""
        return ConfigParser.StoppingCriterion.INDIVIDUAL_EVALUATIONS

    @cfg("Maximum number of evaluations for the search.")
    def max_evaluations(self):
        """Maximum number of evaluations for the search."""
        return 1000

    @cfg("Show progress of the search.")
    def show_progress(self):
        """Show progress of the search."""
        return True

    @cfg("Sigma value for the gaussian noise.")
    def mutation_sigma(self):
        """Sigma value for the gaussian noise."""
        return 50
