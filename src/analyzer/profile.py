import cProfile
from pstats import SortKey, Stats


def profile_it(func):
    def wrapper(*args, **kwargs):
        with cProfile.Profile() as pr:
            result = func(*args, **kwargs)

        Stats(pr).sort_stats(SortKey.CUMULATIVE).print_stats()
        return result

    return wrapper
