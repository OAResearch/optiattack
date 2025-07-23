"""
File is part of the following publication:

Bartlett, A., Liem, C. C., & Panichella, A. (2024).
Multi-objective differential evolution in the generation of adversarial examples.
Science of Computer Programming, 238, 103169.
"""

# src/pixel_change_minimise.py
from abc import ABC

from pymoo.core.problem import ElementwiseProblem

from core.search.action import Action
from core.search.individual import Individual
from core.utils.images import ProcessedImage


# Super class extending Elementwiseproblem, used by all our problems
class PixelChangeMinimise(ElementwiseProblem, ABC):

    """Base class for problems that involve modifying pixel values in an image"""

    def __init__(self,
                 config: dict,
                 objectives=1):
        """Initializes the PixelChangeMinimise problem with configuration and objectives."""
        super().__init__(n_var=3, n_obj=objectives)
        self.rows = config.get("image_height")
        self.cols = config.get("image_width")
        self.image_as_array = None
        self.ff = None
        self.archive = None

    def set_image_as_array(self, image: ProcessedImage):
        """Sets the image as an array for processing."""
        self.image_as_array = image.array

    def set_fitness_function(self, ff):
        """Sets the fitness function to be used for evaluating individuals."""
        self.ff = ff

    def set_archive(self, archive):
        """Sets the archive to store individuals."""
        self.archive = archive

    def get_predictions(self, changes):
        """Calculates the fitness of an individual based on pixel changes."""
        individual = Individual()
        for action in changes:
            individual.add_action(Action((action[0], action[1]), action[2][0], action[2][1], action[2][2]))
        ei = self.ff.calculate_fitness(individual)
        self.archive.add_archive_if_needed(ei)

        return ei
