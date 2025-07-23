"""
File is part of the following publication:

Bartlett, A., Liem, C. C., & Panichella, A. (2024).
Multi-objective differential evolution in the generation of adversarial examples.
Science of Computer Programming, 238, 103169.
"""

# src/de_mutations.py

# Contains all the different mutations for our program.
import math
import random

import numpy as np
from pymoo.core.mutation import Mutation


def contains(row, column, vector):
    """Checks if the given row and column exist in the vector."""
    for index in range(len(vector)):
        if vector[index][0] == row and vector[index][1] == column:
            return index
    return -1


# GuassianMutation - Takes each generation and overwrites a random pixel in each one.
# We return the new seed along with its new fitness.
def add_gaussian_mutation(problem, matrix):
    """Adds a Gaussian mutation to the problem's image."""

    count = 0
    while random.random() <= math.pow(0.5, count):
        # indexes of the pixel to change
        row_index = np.random.randint(0, problem.rows - 1)
        col_index = np.random.randint(0, problem.cols - 1)

        # Create a copy of the seed image
        seed = problem.image_as_array.copy()

        if contains(row_index, col_index, matrix) == -1:
            value, noise = gaussian_noise(seed[row_index][col_index])
            matrix.append([row_index, col_index, value, noise])

        count = count + 1

    return matrix


def calculate_noise(pixel, delta):
    """Calculates the noise introduced by the Gaussian mutation."""
    # This function is currently not used in the code.

    # value = pixel + delta
    # noise = 0
    # Loop through and calculate the total amount of noise on the RGB channel
    # We cap at 0 and 255 so we need to calculate the specific amount
    # Tally across all three channels and return the value
    # for i in range(0, len(delta)):
    #     if value[i] <= 0:
    #         noise += pixel[i]
    #     elif value[i] >= 255:
    #         noise += 255 - pixel[i]
    #     else:
    #         noise += abs(delta[i])

    return 0


def gaussian_noise(pixel):
    """Generates a Gaussian noise value for the pixel."""

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

    return matrix


class PixelMutation(Mutation):

    """PixelMutation - Applies a differential evolution mutation to the pixel values of an image."""

    def __init__(self):
        """Initialize the mutation with specific parameters."""
        super().__init__()
        self.CR = 0.9
        self.F = 0.8

    # gaussian_mutation - Randomly modifies the pixel RGB using the random.gauss function.
    # seed - The seed image we're modifying.
    # matrix - The change matrix we're modifying.

    def _do(self, problem, mutations, **kwargs):
        """Applies the mutation to the given problem and mutations."""

        # for each individual
        for i in range(len(mutations)):
            predictions = mutations[i, 2] if mutations[i, 2] else []
            applied_mutations = mutations[i, 1] if mutations[i, 1] else []
            indexes = np.random.choice(range(len(mutations)), size=3)

            a = mutations[indexes[0], 2] if mutations[indexes[0], 2] else []
            b = mutations[indexes[1], 2] if mutations[indexes[1], 2] else []
            c = mutations[indexes[2], 2] if mutations[indexes[2], 2] else []

            a = [[('', pred.label, pred.value) for pred in _.fitness.predictions] for _ in a]
            b = [[('', pred.label, pred.value) for pred in _.fitness.predictions] for _ in b]
            c = [[('', pred.label, pred.value) for pred in _.fitness.predictions] for _ in c]

            apply_to_index = np.random.randint(0, len(applied_mutations))
            for index in range(len(applied_mutations)):
                p = np.random.random()
                if p < self.CR or index == apply_to_index:
                    donor1 = self.get_pixel_values(applied_mutations[index][0], applied_mutations[index][1], a)
                    donor2 = self.get_pixel_values(applied_mutations[index][0], applied_mutations[index][1], b)
                    donor3 = self.get_pixel_values(applied_mutations[index][0], applied_mutations[index][1], c)

                    new = np.add(donor1, np.multiply(self.F, np.subtract(donor2, donor3)))

                    new[new < 0] = 0
                    new[new > 255] = 255
                    applied_mutations[index][2] = new

            if len(applied_mutations) < 3:
                count = 0
                while random.random() <= math.pow(0.5, count):
                    applied_mutations = add_gaussian_mutation(
                        problem,
                        applied_mutations)
                    count = count + 1
            else:
                if random.random() <= 0.9:
                    count = 0
                    while random.random() <= math.pow(0.5, count):
                        applied_mutations = add_gaussian_mutation(
                            problem,
                            applied_mutations)
                        count = count + 1
                else:
                    applied_mutations = delete_mutation(
                        problem,
                        applied_mutations)

            # Loop through the matrix and update the local seed
            seed = problem.image_as_array.copy()
            for pixel_change in applied_mutations:
                seed[pixel_change[0]][pixel_change[1]] = pixel_change[2]

            new_predictions = problem.get_predictions(applied_mutations)
            predictions.append(new_predictions)

            mutations[i, 0] = seed
            mutations[i, 1] = applied_mutations
            mutations[i, 2] = predictions
            # end for cycle

        return mutations

    def get_pixel_values(self, row, column, donor):
        """Retrieves the pixel values from the donor matrix based on the row and column."""

        pixel_index = contains(row, column, donor)
        if pixel_index == -1:
            A = [0, 0, 0]
        else:
            A = donor[pixel_index][2]
        return A
