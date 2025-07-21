# src/pixel_change_minimise.py
from abc import ABC

from pymoo.core.problem import ElementwiseProblem

from core.search.action import Action
from core.search.individual import Individual
from core.utils.images import ProcessedImage


# Super class extending Elementwiseproblem, used by all our problems
class PixelChangeMinimise(ElementwiseProblem, ABC):
    def __init__(self,
                 config: dict,
                 objectives=1):
        super().__init__(n_var=3, n_obj=objectives)
        self.rows = config.get("image_height")
        self.cols = config.get("image_width")
        self.image_as_array = None
        self.ff = None
        self.archive = None

    def set_image_as_array(self, image: ProcessedImage):
        self.image_as_array = image.array

    def set_fitness_function(self, ff):
        self.ff = ff

    def set_archive(self, archive):
        self.archive = archive

    def get_predictions(self, changes):
        individual = Individual()
        for action in changes:
            individual.add_action(Action((action[0], action[1]), action[2][0], action[2][1], action[2][2]))
        ei = self.ff.calculate_fitness(individual)
        self.archive.add_archive_if_needed(ei)

        return ei