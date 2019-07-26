from time import time


class timeit(object):
    """Minimal logging for timing things."""

    def __init__(self, name):
        self.name = name
        self.t1 = None

    def __enter__(self):
        print("Started '{}'".format(self.name))
        self.name
        self.t1 = time()

    def __exit__(self, a, b, c):
        delta = time() - self.t1
        print("Finished '{}' in {} seconds".format(self.name, delta))
