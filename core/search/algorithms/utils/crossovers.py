# src/crossovers.py
from pymoo.core.crossover import Crossover
import numpy as np
import math
import random
import time



# 4 way crossover where 1 changes all and 3 change sections
# TODO: WIP: This is a WIP code is very dirty right now as I've just been playing with it.
class T4ce(Crossover):
    def __init__(self):
        # define the crossover: number of parents and number of offsprings
        super().__init__(4, 4)

    def _do(self, problem, children, **kwargs):
        # do some magic here
        _, n_matings, n_var = children.shape

        # The output with the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        crossovers = np.full_like(children, None, dtype=object)

        if len(children[0]) == 1:
            return children

        # for each mating provided
        for k in range(n_matings):
            # get a group of 4 children that we'll label a,b,c,d where d is fully deconstructed
            # TODO: Modify to select from a random child column .. so that a,b,c,d aren't always from 1,2,3,4
            a, b, c, d = children[0, k, 2], children[1, k, 2], children[2, k, 2], children[3, k, 2]

            cousins = [a.copy(), b.copy(), c.copy(), d.copy()]

            d_matrix_size = len(cousins[3])
            if d_matrix_size == 2:
                # Because we only need two children when d is too small to split 3 ways
                # Randomly choose 2 children from a,b,c
                child_1 = child_2 = random.randint(0, 2)
                while child_2 == child_1:
                    child_2 = random.randint(1, 3)

                slice1_index = random.randint(0, len(cousins[child_1])-1)
                slice2_index = random.randint(0, len(cousins[child_2])-1)

                cousins[3][0] = cousins[child_1][slice1_index]
                cousins[3][1] = cousins[child_2][slice2_index]
                cousins[child_1][slice1_index] = d[0]
                cousins[child_2][slice2_index] = d[1]

                crossovers[0, k, 2], crossovers[1, k, 2], crossovers[2, k, 2], crossovers[3, k, 2] = a, b, c, d
            if d_matrix_size > 2:
                # We want to ensure that we don't hog the whole array
                d_left = random.randint(0, len(d)-3)
                d_right = random.randint(d_left + 1, len(d)-2)

                # print(f"a: {len(a)} | b: {len(b)} | c: {len(c)} | d: {len(d)}")
                # print(f"d_left: {d_left} | d_right: {d_right}")

                for i in range(0, d_left+1):
                    if len(cousins[0]) > i:
                        # print(f"cousins 0 - len: {len(cousins[0])} | index: {i}")
                        cousins[3][i] = cousins[0][i]
                        cousins[0][i] = d[i]

                diff = d_right - d_left
                for i in range(0, diff):
                    # Get the difference between the two points so we know there's enough room
                    d_index = d_left + 1 + i
                    use_index = d_index
                    if len(cousins[1]) < 2:
                        use_index = 0
                    elif len(cousins[1]) < 3:
                        use_index = 1
                    elif len(cousins[1]) < i:
                        diff_index = len(cousins[1]) - diff
                        use_index = diff_index if diff_index > 0 else 0

                    if len(cousins[1]) > use_index:
                        # print(f"cousins 1 - len: {len(cousins[1])} | index: {use_index}")
                        cousins[3][d_index] = cousins[1][use_index]
                        cousins[1][use_index] = d[d_index]

                diff = len(d)-d_right-1
                for i in range(0, diff):
                    cousin_2_i = len(cousins[2])-diff+i if len(cousins[2])-diff+i > 0 else 0
                    d_i = len(d)-diff+i
                    # print(f"diff: {diff} | cousin_2_i: {cousin_2_i} | d_i: {d_i}")
                    if len(cousins[2]) > cousin_2_i:
                        # print(f"cousins 2 - len: {len(cousins[2])} | index: {i}")
                        cousins[3][d_i] = cousins[2][cousin_2_i]
                        cousins[2][cousin_2_i] = d[d_i]

                crossovers[0, k, 2], crossovers[1, k, 2], crossovers[2, k, 2], crossovers[3, k, 2] = cousins[0], cousins[1], cousins[2], cousins[3]

        return crossovers



# ArraySlice - Takes a chunk of an array that fits both children. For example array section from 3-6;
# then swaps them over to create a new child.
# Currently has a part commented out that attempts to make things more random.
class ArraySlice(Crossover):
    def __init__(self):
        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)

    def _do(self, problem, children, **kwargs):
        # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = children.shape

        # The output with the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        crossovers = np.full_like(children, None, dtype=object)

        if len(children[0]) == 1:
            return children

        # for each mating provided
        for k in range(n_matings):
            # Generate a random index and use this as our a and b crossover
            # a_index = random.randint(0, len(X[0])-1)
            # b_index = random.randint(0, len(X[0])-1)
            # while a_index == b_index:
            #     b_index = random.randint(0, len(X[0])-1)

            # get the first and the second parent
            a, b = children[0, k, 1], children[1, k, 1]

            # prepare the offsprings by copying the change matrix
            off_a = a.copy()
            off_b = b.copy()

            if len(off_a) > 1 and len(off_b) > 1:
                smallest_matrix = off_a if len(off_a) <= len(off_b) else off_b
                start_index = random.randint(1, math.floor(len(smallest_matrix) / 2))
                end_index = random.randint(start_index, len(smallest_matrix)-1)

                for i in range(start_index, end_index):
                    off_a[i] = b[i]
                    off_b[i] = a[i]

            # join the character list and set the output
            crossovers[0, k, 1], crossovers[1, k, 1] = off_a, off_b
            crossovers[0, k, 2], crossovers[1, k, 2] = children[0, k, 2], children[1, k, 2]

        return crossovers
