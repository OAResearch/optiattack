# src/sampling.py
from pymoo.core.sampling import Sampling
import numpy as np

from core.search.algorithms.utils.mutations import add_gaussian_mutation


# InitialImageSampling - Generates our base samples. Uses the given image_array to generate a new seed.
# Modifies one pixel in this seed and adds it to our sample array along with the initial prediction value.
class InitialImageSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        samples = np.full((n_samples, 3), None, dtype=object)

        for i in range(n_samples):
            seed, matrix = add_gaussian_mutation(problem, [])
            predictions = problem.get_predictions(matrix)

            samples[i, 0] = seed
            samples[i, 1] = matrix
            samples[i, 2] = [predictions]

        return samples
