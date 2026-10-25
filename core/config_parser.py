"""Configuration file for the Client."""
import argparse
import inspect
import logging
from typing import Any

from core.utils.decorators import cfg


def t_or_f(arg):
    """Convert string to boolean."""
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
        return True
    elif 'FALSE'.startswith(ua):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected for {arg}")


def nullable_string(val):
    """Convert string to None or string."""
    if not val:
        return None
    return val


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
        if type(default) is bool:
            self.parser.add_argument(f"--{name}",
                                     default=default,
                                     help=description,
                                     type=t_or_f,
                                     **kwargs)
        elif name == "target":
            self.parser.add_argument(f"--{name}",
                                     default=default,
                                     help=description,
                                     type=nullable_string,
                                     **kwargs)
        else:
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

    def to_markdown(self, output_file: str = "../docs/parameters.md"):
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
        if args.attack_type == ConfigParser.AttackType.TARGETED and args.target is None:
            raise ValueError("Target class is only valid for targeted attacks.")

        if args.attack_type == ConfigParser.AttackType.UNTARGETED and args.target is not None:
            logging.warning("Target class is not valid for untargeted attacks. Ignoring it.")

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
        return None

    class AttackType:

        """Attack type for the search."""

        TARGETED = "targeted"
        UNTARGETED = "untargeted"

    @cfg("Attack type for the search. Options: 'targeted' or 'untargeted'.")
    def attack_type(self):
        """Attack type for the search."""
        return ConfigParser.AttackType.UNTARGETED

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
        GENETIC = "genetic"
        ABC = "abc"
        MABC = "mabc"
        BASELINE = "baseline"

    @cfg("Search algorithm for the optimization.")
    def algorithm(self):
        """Search algorithm for the optimization."""
        return ConfigParser.Algorithms.MIO

    @cfg("Population size of the population-based algorithms.")
    def population_size(self):
        """Population size of the population-based algorithms."""
        return 10

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

    @cfg("Block size for shuffle mutator. Must be odd number (e.g., 3, 5, 7).")
    def shuffle_block_size(self):
        """Block size for shuffle mutator."""
        return 3

    @cfg("Snapshot interval for the search.")
    def snapshot_interval(self):
        """Snapshot interval for the search. If set -1 no snapshots are saved."""
        return 5

    @cfg("Path to the output directory.")
    def output_dir(self):
        """Path to the output directory."""
        return "./output"

    @cfg("Final image name.")
    def final_image_name(self):
        """Final image name."""
        return "final_image"

    @cfg("Overlay matrix image name.")
    def overlay_image_name(self):
        """Overlay matrix image name."""
        return "overlay_matrix"

    @cfg("Line plot image name.")
    def line_plot_name(self):
        """Line plot image name."""
        return "line_plot"

    @cfg("Statistics file name.")
    def statistics_file_name(self):
        """Statistics file name."""
        return "statistics"

    @cfg("Write the statistics to a file.")
    def write_statistics(self):
        """Write the results to a file."""
        return True

    @cfg("Save the generated images.")
    def save_images(self):
        """Save the generated images."""
        return True

    @cfg("Statistics column id.")
    def statistics_column_id(self):
        """Specifies the identifier of the column in the statistics file to be used for analysis or plotting."""
        return "-"

    @cfg("Experiment label.")
    def experiment_label(self):
        """Experiment label."""
        return "experiment"

    @cfg("Show plots.")
    def show_plots(self):
        """Show plots."""
        return True

    class Crossovers:

        """Crossover operators for the search."""

        SINGLE_POINT_CROSSOVER = "single_point_crossover"

    @cfg("Crossover operator for the search.")
    def crossover(self):
        """Mutation operator for the search."""
        return ConfigParser.Crossovers.SINGLE_POINT_CROSSOVER

    class Mutators:

        """Mutation operators for the search."""

        STANDARD_MUTATOR = "gaussian_mutator"
        ONE_ZERO_MUTATOR = "one_zero_mutator"
        SHUFFLE_MUTATOR = "shuffle_mutator"

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
        GAUSSIAN_SAMPLER = "gaussian_sampler"
        SHUFFLE_SAMPLER = "shuffle_sampler"

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

    @cfg("Enable pruning of the final results.")
    def enable_pruning(self):
        """Enable pruning of the final results."""
        return False

    class PruningTypes:

        """Pruning methods for the search."""

        STANDARD = "standard"

    @cfg("Pruning method for the search.")
    def pruning_method(self):
        """Pruning method for the search."""
        return ConfigParser.PruningTypes.STANDARD

    @cfg("Abandon limit for ABC algorithm. Number of trials before a food source is abandoned.")
    def abc_limit(self):
        """Abandon limit for ABC algorithm."""
        return 10

    @cfg("Fitness threshold for stopping the search. Search stops when fitness <= this value. "
         "Default is 0.0 (stop when solution is found). "
         "Use negative values for stricter stopping criteria (e.g., -0.1).")
    def fitness_threshold(self):
        """Fitness threshold for stopping the search."""
        return 0.0

    @cfg("ART baseline attack used by the 'baseline' algorithm. Options: 'square', 'hsj', 'simba', 'pixel'.")
    def baseline_attack(self):
        """ART baseline attack used by the 'baseline' algorithm."""
        return "square"

    @cfg("Stop the baseline attack at the first successful query. 'auto' uses the attack default (square, "
         "simba, pixel: true; hsj: false, it uses the whole budget and reports the smallest successful "
         "perturbation). Options: 'auto', 'true', 'false'.")
    def baseline_stop_at_success(self):
        """Stop the baseline attack at the first successful query."""
        return "auto"

    @cfg("Norm of the Square attack. Options: 'inf' or '2'.")
    def square_norm(self):
        """Norm of the Square attack."""
        return "inf"

    @cfg("Maximum perturbation of the Square attack on the 0-255 scale (ART default 0.3 is on 0-1).")
    def square_eps(self):
        """Maximum perturbation of the Square attack on the 0-255 scale (ART default 0.3 is on 0-1)."""
        return 8.0

    @cfg("Initial fraction of the image changed by a Square attack update (ART default).")
    def square_p_init(self):
        """Initial fraction of the image changed by a Square attack update (ART default)."""
        return 0.8

    @cfg("Number of restarts of the Square attack (ART default).")
    def square_nb_restarts(self):
        """Number of restarts of the Square attack (ART default)."""
        return 1

    @cfg("Iteration limit of the Square attack; its square size schedule is relative to it. "
         "0 (auto): max_evaluations for the 'individual_evaluations' criterion, 10000 for 'time'.")
    def square_max_iter(self):
        """Iteration limit of the Square attack (0: auto)."""
        return 0

    @cfg("Norm of the HopSkipJump attack. Options: '2' or 'inf'.")
    def hsj_norm(self):
        """Norm of the HopSkipJump attack."""
        return "2"

    @cfg("Maximum number of iterations of the HopSkipJump attack (ART default).")
    def hsj_max_iter(self):
        """Maximum number of iterations of the HopSkipJump attack (ART default)."""
        return 50

    @cfg("Maximum number of evaluations for gradient estimation in HopSkipJump (ART default).")
    def hsj_max_eval(self):
        """Maximum number of evaluations for gradient estimation in HopSkipJump (ART default)."""
        return 10000

    @cfg("Initial number of evaluations for gradient estimation in HopSkipJump (ART default).")
    def hsj_init_eval(self):
        """Initial number of evaluations for gradient estimation in HopSkipJump (ART default)."""
        return 100

    @cfg("Maximum number of random trials for the initial adversarial image of HopSkipJump (ART default).")
    def hsj_init_size(self):
        """Maximum number of random trials for the initial adversarial image of HopSkipJump (ART default)."""
        return 100

    @cfg("Path to an image used as the initial adversarial image of HopSkipJump (e.g. an image of the target "
         "class). Empty: random initialisation.")
    def hsj_init_image(self):
        """Path to an image used as the initial adversarial image of HopSkipJump."""
        return ""

    @cfg("SimBA basis. Options: 'px' or 'dct' (ART default; its steps are spread over the whole image and can "
         "vanish when rounded to uint8 on large images unless simba_epsilon is large).")
    def simba_attack(self):
        """Basis of the SimBA attack."""
        return "px"

    @cfg("SimBA step size on the 0-255 scale (ART default 0.1 is on 0-1).")
    def simba_epsilon(self):
        """Step size of the SimBA attack on the 0-255 scale."""
        return 25.5

    @cfg("Iteration limit of SimBA. High so that max_evaluations is the real limit.")
    def simba_max_iter(self):
        """Iteration limit of SimBA."""
        return 100000

    @cfg("SimBA order of the basis directions. Options: 'random' or 'diag' (ART default 'random').")
    def simba_order(self):
        """Order of the basis directions of the SimBA attack."""
        return "random"

    @cfg("SimBA low frequency dimension for the 'dct' basis (ART default).")
    def simba_freq_dim(self):
        """Low frequency dimension of the SimBA 'dct' basis."""
        return 4

    @cfg("SimBA stride for the 'dct' basis with 'diag' order (ART default).")
    def simba_stride(self):
        """Stride of the SimBA 'dct' basis."""
        return 1

    @cfg("Number of pixels changed by the PixelAttack.")
    def pixel_th(self):
        """Number of pixels changed by the PixelAttack."""
        return 10

    @cfg("Optimiser of the PixelAttack. 0: CMA-ES, 1: differential evolution (ART default).")
    def pixel_es(self):
        """Optimiser of the PixelAttack."""
        return 1

    @cfg("Maximum number of optimiser iterations of the PixelAttack (ART default).")
    def pixel_max_iter(self):
        """Maximum number of optimiser iterations of the PixelAttack (ART default)."""
        return 100


if __name__ == "__main__":
    config_parser = ConfigParser()
    config = config_parser.parse_args()
    config_parser.to_markdown()
