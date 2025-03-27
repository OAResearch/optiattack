from io import BytesIO

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from PIL import Image

from core.search.action import Action
from core.search.evaluated_individual import EvaluatedIndividual
from core.search.fitness_value import FitnessValue
from core.search.individual import Individual
from core.search.service.randomness import Randomness
from core.search.service.search_time_controller import SearchTimeController
from core.utils.images import ProcessedImage
from core.search.service.archive import Archive


# Fixture for the Archive instance
@pytest.fixture
def archive():
    randomness = MagicMock(spec=Randomness)
    config = {"image_width": 100, "image_height": 100}
    stc = SearchTimeController(config)
    return Archive(stc, randomness, config)

# Test cases
def test_clean_population(archive):
    # Add some dummy populations
    archive.populations = [MagicMock(), MagicMock()]
    archive.clean_population()
    assert len(archive.populations) == 0

def test_add_archive_if_needed_with_better_fitness(archive):
    individual = Individual()
    individual.get_actions = MagicMock(return_value=[MagicMock(spec=Action)])
    fitness = MagicMock(spec=FitnessValue)
    fitness.value = 0.6

    ei1 = EvaluatedIndividual(individual, fitness)

    individual2 = Individual()
    individual2.get_actions = MagicMock(return_value=[MagicMock(spec=Action)])
    fitness2 = MagicMock(spec=FitnessValue)
    fitness2.value = 0.5

    ei2 = EvaluatedIndividual(individual2, fitness2)
    archive.stc.set_current_fitness(ei1.fitness)

    archive.add_archive_if_needed(ei2)

    # Assert that the individual was added to the populations
    assert len(archive.populations) == 1
    assert archive.stc.get_current_fitness_value() == 0.5

def test_add_archive_if_needed_with_worse_fitness(archive):
    individual = Individual()
    individual.get_actions = MagicMock(return_value=[MagicMock(spec=Action)])
    fitness = MagicMock(spec=FitnessValue)
    fitness.value = 0.6

    ei1 = EvaluatedIndividual(individual, fitness)

    individual2 = Individual()
    individual2.get_actions = MagicMock(return_value=[MagicMock(spec=Action)])
    fitness2 = MagicMock(spec=FitnessValue)
    fitness2.value = 0.7

    ei2 = EvaluatedIndividual(individual2, fitness2)
    archive.stc.set_current_fitness(ei1.fitness)

    archive.add_archive_if_needed(ei2)

    # Assert that the individual was not added to the populations
    assert len(archive.populations) == 0

def test_is_empty(archive):
    # Test when the archive is empty
    assert archive.is_empty()

    # Test when the archive is not empty
    archive.populations = [MagicMock()]
    assert not archive.is_empty()

def test_number_of_population(archive):
    # Test when the archive is empty
    assert archive.number_of_population() == 0

    # Test when the archive has populations
    archive.populations = [MagicMock(), MagicMock()]
    assert archive.number_of_population() == 2

def test_set_and_get_image(archive):
    # Create a mock image
    image = MagicMock(spec=ProcessedImage)

    # Set the image
    archive.set_image(image)

    # Get the image and assert it is the same
    assert archive.get_image() == image

def test_get_actions(archive):
    # Mock populations with actions
    action1 = MagicMock(spec=Action)
    action2 = MagicMock(spec=Action)
    ei1 = MagicMock(spec=EvaluatedIndividual)
    ei1.individual = MagicMock(spec=Individual)
    ei1.individual.get_actions.return_value = [action1]

    ei2 = MagicMock(spec=EvaluatedIndividual)
    ei2.individual = MagicMock(spec=Individual)
    ei2.individual.get_actions.return_value = [action2]
    archive.populations = [ei1, ei2]

    # Get the actions
    actions = archive.get_actions()

    # Assert that the actions are correct
    assert actions == [action1, action2]

def test_get_mutated_image(archive):
    # Mock the image and actions
    image_data = BytesIO()
    image = Image.new("RGB", (10, 10), color="red")
    image.save(image_data, format="JPEG")
    image_data.seek(0)

    image_array = np.array(image)
    archive.image = MagicMock(spec=ProcessedImage)
    archive.image.array = image_array

    action1 = MagicMock(spec=Action)
    action1.get_location.return_value = (0, 1)
    action1.get_color.return_value = np.array([255, 255, 255])
    action2 = MagicMock(spec=Action)
    action2.get_location.return_value = (1, 2)
    action2.get_color.return_value = np.array([128, 128, 128])
    archive.populations = [
        MagicMock(individual=MagicMock(get_actions=MagicMock(return_value=[action1]))),
        MagicMock(individual=MagicMock(get_actions=MagicMock(return_value=[action2])))
    ]

    # Get the mutated image
    mutated_image = archive.get_mutated_image()

    # Assert that the image was mutated correctly
    assert all(mutated_image[0][1] == [255, 255, 255])
    assert all(mutated_image[1][2] == [128, 128, 128])

def test_archive_extract(archive):
    # Mock the actions
    action1 = MagicMock(spec=Action)
    action1.get_location.return_value = (0, 0)
    action1.get_color.return_value = np.array([255, 255, 255])

    action2 = MagicMock(spec=Action)
    action2.get_location.return_value = (1, 1)
    action2.get_color.return_value = np.array([128, 128, 128])

    action3 = MagicMock(spec=Action)
    action3.get_location.return_value = (0, 0)
    action3.get_color.return_value = np.array([42, 42, 42])

    ei1 = MagicMock(spec=EvaluatedIndividual)
    ei1.individual = MagicMock(spec=Individual)
    ei1.individual.get_actions.return_value = [action1]

    ei2 = MagicMock(spec=EvaluatedIndividual)
    ei2.individual = MagicMock(spec=Individual)
    ei2.individual.get_actions.return_value = [action2]

    ei3 = MagicMock(spec=EvaluatedIndividual)
    ei3.individual = MagicMock(spec=Individual)
    ei3.individual.get_actions.return_value = [action3]

    archive.populations = [ei1, ei2, ei3]

    # Extract the solution
    solution = archive.extract_solution()
    assert solution.actions == [action2, action3]


