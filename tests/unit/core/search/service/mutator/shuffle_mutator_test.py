"""Tests for ShuffleMutator class."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from core.search.action import Action
from core.search.individual import Individual
from core.search.phase_controller import PhaseController
from core.search.service.adaptive_parameter_control import AdaptiveParameterControl
from core.search.service.archive import Archive
from core.search.service.mutator.shuffle_mutator import ShuffleMutator
from core.search.service.randomness import Randomness
from core.search.service.search_time_controller import SearchTimeController
from core.utils.images import ProcessedImage


@pytest.fixture
def shuffle_mutator():
    """Create a ShuffleMutator instance with a 6x6 test image."""
    time = MagicMock()
    config = {
        "seed": 42,
        "image_height": 6,
        "image_width": 6,
        "mutation_sigma": 0.1,
        "apc_pixel_start": 0,
        "apc_pixel_end": 255,
        "start_time": 0.5,
        "threshold": 1.0
    }
    randomness = Randomness(config)
    stc = SearchTimeController(config, pc=PhaseController())
    apc = AdaptiveParameterControl(stc, config)

    # Create a 6x6 RGB image with known pixel values
    image_array = np.zeros((6, 6, 3), dtype=np.uint8)
    for i in range(6):
        for j in range(6):
            image_array[i, j] = [i * 40, j * 40, (i + j) * 20]

    processed_image = ProcessedImage(original=None, resized=None, array=image_array)
    archive = Archive(stc, randomness, config)
    archive.set_image(processed_image)

    mutator = ShuffleMutator(randomness, time, config, apc)
    mutator.archive = archive

    return mutator


# Test cases
def test_mutate_with_empty_actions(shuffle_mutator):
    """Test mutate method with an individual that has no actions."""
    individual = Individual()
    result = shuffle_mutator.mutate(individual)

    # ShuffleMutator creates a 3x3 block when empty (unlike StandardMutator)
    assert len(result.get_actions()) == 9


def test_mutate_with_non_empty_actions(shuffle_mutator):
    """Test mutate method with an individual that has actions at (2,2)."""
    original_image = shuffle_mutator.archive.image.array.copy()

    # Create individual with 3x3 block at center (2, 2)
    individual = Individual()
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            row, col = 2 + dr, 2 + dc
            r, g, b = original_image[row, col]
            action = Action(location=(row, col), red=int(r), green=int(g), blue=int(b))
            individual.add_action(action)

    # Mock location_apc to prevent position shift
    shuffle_mutator.apc.get_location_apc = MagicMock(return_value=0)

    result = shuffle_mutator.mutate(individual)

    # Assert that result has 9 actions
    assert len(result.get_actions()) == 9


def test_shuffle_mutation_permutes_colors(shuffle_mutator):
    """Test that shuffle mutation permutes the original block colors."""
    original_image = shuffle_mutator.archive.image.array.copy()

    # Get original 3x3 block colors at center (2, 2)
    original_colors = set()
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            row, col = 2 + dr, 2 + dc
            original_colors.add(tuple(original_image[row, col]))

    # Create individual with actions at center (2, 2)
    individual = Individual()
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            row, col = 2 + dr, 2 + dc
            r, g, b = original_image[row, col]
            action = Action(location=(row, col), red=int(r), green=int(g), blue=int(b))
            individual.add_action(action)

    # Use zero sigma so block doesn't shift
    shuffle_mutator.apc.get_location_apc = MagicMock(return_value=0)

    result = shuffle_mutator.mutate(individual)
    result_colors = set(tuple(a.get_color()) for a in result.get_actions())

    # Colors should be the same set (shuffle is a permutation)
    assert result_colors == original_colors


def test_shuffle_mutation_preserves_and_permutes_pixels(shuffle_mutator):
    """
    Test that shuffle mutation preserves the original pixel values
    but permutes their positions within the 3x3 block.
    """
    original_image = shuffle_mutator.archive.image.array.copy()

    # Create individual with 3x3 block centered at (2, 2)
    individual = Individual()
    center_row, center_col = 2, 2

    # Store original pixels before mutation
    original_pixels = {}
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            row, col = center_row + dr, center_col + dc
            r, g, b = original_image[row, col]
            original_pixels[(row, col)] = (int(r), int(g), int(b))
            action = Action(location=(row, col), red=int(r), green=int(g), blue=int(b))
            individual.add_action(action)

    # Prevent block from shifting position
    shuffle_mutator.apc.get_location_apc = MagicMock(return_value=0)

    # Apply mutation
    result = shuffle_mutator.mutate(individual)

    # Extract mutated pixels
    mutated_pixels = {}
    for action in result.get_actions():
        loc = action.get_location()
        color = action.get_color()
        if isinstance(color, np.ndarray):
            color = tuple(color)
        mutated_pixels[loc] = color

    # Verify: same color set, but different arrangement
    original_colors = set(original_pixels.values())
    mutated_colors = set(mutated_pixels.values())

    assert original_colors == mutated_colors, "Colors must be preserved"

    # Check that at least some positions changed
    changed_positions = sum(1 for loc in original_pixels
                            if original_pixels[loc] != mutated_pixels[loc])
    assert changed_positions > 0, "Shuffle should change at least some pixel positions"


# ============== PARAMETRIC BLOCK SIZE TESTS ==============

def create_mutator_with_block_size(block_size: int):
    """Create a ShuffleMutator with a specific block size and appropriately sized image."""
    time = MagicMock()
    # Image size should be large enough to accommodate the block
    image_size = max(10, block_size * 3)
    config = {
        "seed": 42,
        "image_height": image_size,
        "image_width": image_size,
        "mutation_sigma": 0.1,
        "apc_pixel_start": 0,
        "apc_pixel_end": 255,
        "start_time": 0.5,
        "threshold": 1.0,
        "shuffle_block_size": block_size,  # Parametric block size
    }
    randomness = Randomness(config)
    stc = SearchTimeController(config, pc=PhaseController())
    apc = AdaptiveParameterControl(stc, config)

    # Create RGB image with unique pixel values
    image_array = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    for i in range(image_size):
        for j in range(image_size):
            image_array[i, j] = [(i * 17) % 256, (j * 23) % 256, ((i + j) * 13) % 256]

    processed_image = ProcessedImage(original=None, resized=None, array=image_array)
    archive = Archive(stc, randomness, config)
    archive.set_image(processed_image)

    mutator = ShuffleMutator(randomness, time, config, apc)
    mutator.archive = archive

    return mutator, image_size


@pytest.mark.parametrize("block_size", [2, 3, 4, 5, 6])
def test_parametric_block_size_creates_correct_action_count(block_size):
    """Test that mutator creates correct number of actions for different block sizes."""
    mutator, _ = create_mutator_with_block_size(block_size)
    
    individual = Individual()
    result = mutator.mutate(individual)
    
    expected_actions = block_size * block_size
    assert len(result.get_actions()) == expected_actions, \
        f"Block size {block_size}x{block_size} should create {expected_actions} actions"





def test_block_size_2x2():
    """Test specifically for 2x2 block size."""
    mutator, _ = create_mutator_with_block_size(2)
    
    individual = Individual()
    result = mutator.mutate(individual)
    
    assert len(result.get_actions()) == 4, "2x2 block should have 4 actions"


def test_block_size_6x6():
    """Test specifically for 6x6 block size (the failing case from bug report)."""
    mutator, _ = create_mutator_with_block_size(6)
    
    individual = Individual()
    result = mutator.mutate(individual)
    
    assert len(result.get_actions()) == 36, "6x6 block should have 36 actions"


def test_get_block_size_helper():
    """Test that _get_block_size returns correct value from config."""
    mutator, _ = create_mutator_with_block_size(5)
    assert mutator._get_block_size() == 5
    
    mutator2, _ = create_mutator_with_block_size(6)
    assert mutator2._get_block_size() == 6


def test_is_odd_block_helper():
    """Test that _is_odd_block correctly identifies odd/even block sizes."""
    mutator_odd, _ = create_mutator_with_block_size(5)
    assert mutator_odd._is_odd_block() is True
    
    mutator_even, _ = create_mutator_with_block_size(6)
    assert mutator_even._is_odd_block() is False
