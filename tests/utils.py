import random
from functools import wraps


def run_multiple(times=1000):
    """
    Run a test function multiple times with different seeds.
    This is useful for testing the robustness of the function.

    Args:
        times (int): Number of the runs (default: 1000)
    """

    def decorator(test_func):
        @wraps(test_func)
        def wrapper(*args, **kwargs):
            original_seed = random.getstate()

            for i in range(times):
                try:
                    random.seed(i)
                    test_func(*args, **kwargs)
                except AssertionError as e:
                    raise AssertionError(f"Test failed on iteration {i} with seed {i}. Error: {str(e)}")
                finally:
                    random.setstate(original_seed)

        return wrapper

    return decorator