"""
File is part of the following publication:

Bartlett, A., Liem, C. C., & Panichella, A. (2024).
Multi-objective differential evolution in the generation of adversarial examples.
Science of Computer Programming, 238, 103169.
"""

# src/mutations.py

# Contains all the different mutations for our program.
import math

from pymoo.core.mutation import Mutation
import numpy as np
import random


# GuassianMutation - Takes each generation and overwrites a random pixel in each one.
# We return the new seed along with its new fitness.
def add_gaussian_mutation(problem, matrix):
    """Adds a Gaussian mutation to the seed image."""

    # indexes of the pixel to change
    row_index = random.randint(0, problem.rows - 1)
    col_index = random.randint(0, problem.cols - 1)

    # Create a copy of the seed image
    seed = problem.image_as_array.copy()

    already_changed = False
    existing_mutation = []
    for mutation in matrix:
        if matrix[0] == row_index and matrix[1] == col_index:
            already_changed = True
            existing_mutation = mutation
            break

    if already_changed:
        value, noise = gaussian_noise(existing_mutation[2])
        matrix.remove(existing_mutation)
    else:
        value, noise = gaussian_noise(seed[row_index][col_index])

    # Loop through the matrix and update the local seed
    matrix.append([row_index, col_index, value, noise])
    # for pixel_change in matrix:
    #     seed[pixel_change[0]][pixel_change[1]] = pixel_change[2]

    return seed, matrix


def calculate_noise(pixel, delta):
    """Calculates the noise introduced by the mutation."""
    # This function is currently not used in the code.

    # value = pixel + delta
    # noise = 0
    # # Loop through and calculate the total amount of noise on the RGB channel
    # # We cap at 0 and 255 so we need to calculate the specific amount
    # # Tally across all three channels and return the value
    # for i in range(0, len(delta)):
    #     if value[i] <= 0:
    #         noise += pixel[i]
    #     elif value[i] >= 255:
    #         noise += 255 - pixel[i]
    #     else:
    #         noise += abs(delta[i])

    return 0.0


def gaussian_noise(pixel):
    """Generates Gaussian noise for a pixel."""

    delta = np.round(random.gauss(pixel * 0, 50))

    # the next if condition guarantees that at least a +1/-1 change has been made
    if random.random() <= 0.5:
        delta[delta == 0] = 1
    else:
        delta[delta == 0] = -1

    # index = random.randint(0, 2)
    # pixel[index] = pixel[index] + delta[index]
    value = pixel + delta

    # checking that the new value for the pixel is in [0; 255]
    value[value < 0] = 0
    value[value > 255] = 255

    noise = calculate_noise(pixel, delta)

    return value, noise


def delete_mutation(problem, matrix):
    """Deletes a random mutation from the matrix."""

    # In case matrix is empty
    if len(matrix) >= 1:
        matrix.pop(random.randint(0, len(matrix) - 1))

    # Loop through the matrix and update the local seed
    seed = problem.image_as_array.copy()
    for pixel_change in matrix:
        seed[pixel_change[0]][pixel_change[1]] = pixel_change[2]

    return seed, matrix


class PixelMutation(Mutation):

    """PixelMutation - Applies Gaussian mutations to the pixel RGB values of the seed image."""

    def __init__(self):
        """Initializes the PixelMutation class."""
        super().__init__()

    # gaussian_mutation - Randomly modifies the pixel RGB using the random.gauss function.
    # seed - The seed image we're modifying.
    # matrix - The change matrix we're modifying.

    def _do(self, problem, mutations, **kwargs):
        """Applies mutations to the problem's seed images."""

        # for each individual
        for i in range(len(mutations)):
            applied_mutations = mutations[i, 2] if mutations[i, 2] else []

            if len(applied_mutations) < 5:
                count = 0
                while random.random() <= math.pow(0.5, count):
                    seed, matrix = add_gaussian_mutation(
                        problem,
                        applied_mutations)
                    count = count + 1
            else:
                p = random.random()
                if p <= 0.9:
                    count = 0
                    while random.random() <= math.pow(0.5, count):
                        seed, matrix = add_gaussian_mutation(
                            problem,
                            applied_mutations)
                        count = count + 1
                else:
                    seed, matrix = delete_mutation(
                        problem,
                        applied_mutations)

            values = problem.get_predictions(seed)
            fitness = values[0][2]
            mutations[i, 0] = fitness
            mutations[i, 1] = seed
            mutations[i, 2] = matrix
            # end for cycle

        return mutations
