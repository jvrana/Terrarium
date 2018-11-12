from collections import Counter
import functools


class HashCounter(Counter):

    def __init__(self, data=(), func=None):
        self.hash_function = func
        super().__init__()
        for d in data:
            self[d] += 1

    def by_attrs(self, *attrs):
        self.hash_function = functools.partial(self.hash_by_attributes, attributes=attrs)
        return self

    @staticmethod
    def hash_by_attributes(data, attributes=()):
        return "%".join([str(getattr(data, x, None)) for x in attributes])

    def __setitem__(self, k, v):
        return super().__setitem__(self.hash_function(k), v)

    def __getitem__(self, k):
        return super().__getitem__(self.hash_function(k))


class HashView(object):

    def __init__(self, data):
        self.data = data
        self.counters = {}

    def register(self, name, function):
        new_counter = HashCounter(self.data, func=function)
        self.counters[name] = new_counter
        return new_counter

    def __getitem__(self, key):
        return self.counters[key]

    def __setitem__(self, key, val):
        return self.register(key, val)
