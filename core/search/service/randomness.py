import numpy as np


class Randomness:
    def __init__(self, config):
        self.random = np.random
        self.update_seed(config.get("seed"))

    # A negative value means the current CPU time clock is used instead
    def update_seed(self, seed):
        if seed < 0:
            self.random.seed()
        else:
            self.random.seed(seed)

    def next_float(self, min_value=None, max_value=None):
        if min_value is None and max_value is None:
            return self.random.uniform(0, 1)
        elif min_value is None:
            return self.random.uniform(0, max_value)
        elif max_value is None:
            return self.random.uniform(min_value, 1)
        return self.random.uniform(min_value, max_value)

    def next_bool(self, value):
        return self.random.rand() < value

    def next_int(self, min_value=None, max_value=None):
        if min_value is None and max_value is None:
            return self.random.randint(0, 1)
        elif min_value is None:
            return self.random.randint(0, max_value)
        elif max_value is None:
            return self.random.randint(min_value, 1)
        return self.random.randint(min_value, max_value)

    def random_choice(self, max_size, selection_probs=None):
        if selection_probs is None:
            return self.random.choice(max_size)
        else:
            return self.random.choice(max_size, p=selection_probs)

    def random_gaussian(self, mean, std):
        return self.random.normal(mean, std)
