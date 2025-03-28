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
        return "./tests/test_img.jpeg"

    @cfg("Image width in pixels. Should be same as the model input size.")
    def image_width(self):
        """Image width in pixels."""
        return 224

    @cfg("Image height in pixels. Should be same as the model input size.")
    def image_height(self):
        """Image height in pixels."""
        return 224

    @cfg("Target class for the targeted attack. If not specified, any misclassification is considered successful.")
    def target(self):
        """Target class for the targeted attack."""
        return "None"

    class StoppingCriterion:

        """Stopping criterion for the search."""

        INDIVIDUAL_EVALUATIONS = "individual_evaluations"
        TIME = "time"

    @cfg("Stopping criterion for the search. "
         "Options: 'individual_evaluations' or 'time'.")
    def stopping_criterion(self):
        """Stopping criterion for the search."""
        return ConfigParser.StoppingCriterion.INDIVIDUAL_EVALUATIONS

    class Algorithms:

        """Search algorithms for the optimization."""

        RANDOM_SEARCH = "random"
        MIO = "mio"

    @cfg("Search algorithm for the optimization.")
    def algorithm(self):
        """Search algorithm for the optimization."""
        return ConfigParser.Algorithms.MIO

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

    @cfg("Snapshot interval for the search.")
    def snapshot_interval(self):
        """Snapshot interval for the search. If set -1 no snapshots are saved."""
        return 5

    @cfg("Path to the output directory.")
    def output_dir(self):
        """Path to the output directory."""
        return "./output"

    @cfg("Write the statistics to a file.")
    def write_statistics(self):
        """Write the results to a file."""
        return True

    @cfg("Save the generated images.")
    def save_images(self):
        """Save the generated images."""
        return True

    @cfg("Experiment label.")
    def experiment_label(self):
        """Experiment label."""
        return "experiment"

    @cfg("Show plots.")
    def show_plots(self):
        """Show plots."""
        return True

    class Mutators:

        """Mutation operators for the search."""

        STANDARD_MUTATOR = "gaussian_mutator"
        ONE_ZERO_MUTATOR = "one_zero_mutator"

    @cfg("Mutation operator for the search.")
    def mutator(self):
        """Mutation operator for the search."""
        return ConfigParser.Mutators.ONE_ZERO_MUTATOR

    @cfg("Mutation rate for the one mutation.")
    def one_mutation_rate(self):
        """Mutation rate for the one mutation."""
        return 0.3

    @cfg("Mutation rate for the zero mutation.")
    def zero_mutation_rate(self):
        """Mutation rate for the zero mutation."""
        return 0.4

    @cfg("Minimum action size")
    def min_action_size(self):
        """Minimum action size."""
        return 1

    @cfg("Maximum action size")
    def max_action_size(self):
        """Maximum action size."""
        return 10

    class SamplerType:

        """Sampler type for the search."""

        RANDOM_SAMPLER = "random_sampler"

    @cfg("Sampler type for the search.")
    def sampler(self):
        """Sampler type for the search."""
        return ConfigParser.SamplerType.RANDOM_SAMPLER

    @cfg("The start value of the adaptive parameter control")
    def apc_start_time(self):
        """APC start time."""
        return 0.4

    @cfg("The threshold value of the adaptive parameter control")
    def apc_threshold(self):
        """APC threshold."""
        return 0.6

    @cfg("The start value of the adaptive parameter control for the pixel value")
    def apc_pixel_start(self):
        """APC pixel start value."""
        return 40

    @cfg("The end value of the adaptive parameter control for the pixel value")
    def apc_pixel_end(self):
        """APC pixel end value."""
        return 30

    @cfg("The start value of the adaptive parameter control for the location")
    def apc_location_start(self):
        """APC noise start value."""
        return 40

    @cfg("The end value of the adaptive parameter control for the location")
    def apc_location_end(self):
        """APC noise end value."""
        return 30

    @cfg("The percentage of passed search before starting a more focused, less exploratory one")
    def focused_search_activation_time(self):
        """Focused search activation time."""
        return 0.8

    @cfg("Probability of sampling a new individual at random")
    def random_sampling_probability(self):
        """Random search probability."""
        return 0.5

    @cfg("Enable web interface.")
    def enable_ui(self):
        """Enable web interface."""
        return False

    class PruningTypes:

        """Pruning methods for the search."""

        STANDARD = "standard"

    @cfg("Pruning method for the search.")
    def pruning_method(self):
        """Pruning method for the search."""
        return ConfigParser.PruningTypes.STANDARD

    @cfg("Enable pruning of the final results.")
    def enable_pruning(self):
        """Enable pruning of the final results."""
        return True
