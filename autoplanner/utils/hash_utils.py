from collections import Counter


class HashCounter(object):
    """
    HashCounter counts objects according to a custom hashing function. A superficial composition
    of a Counter instance
    """

    def __init__(self, func, data=None):
        self.counter = Counter()
        self.hash_function = func
        if data:
            for d in data:
                self[d] += 1

    @staticmethod
    def hash_by_attributes(data, attributes=()):
        return "%".join([str(getattr(data, x, None)) for x in attributes])

    def clear(self):
        self.counter.clear()

    def copy(self):
        return self.__copy__()

    def __mul__(self, num):
        new = self.copy()
        for k, v in new.counter.items():
            new.counter[k] = v * num
        return new

    def __add__(self, other):
        new = self.copy()
        new.counter = self.counter + other.counter
        return new

    def __sub__(self, other):
        new = self.copy()
        new.counter = self.counter - other.counter
        return new

    def __iter__(self):
        return self.counter.__iter__()

    def __setitem__(self, k, v):
        self.counter[self.hash_function(k)] = v

    def __getitem__(self, k):
        return self.counter[self.hash_function(k)]

    def __copy__(self):
        new = self.__class__(self.hash_function)
        new.counter = self.counter.copy()
        return new


class HashView(object):
    """
    HashView maintains a named dictionary of HashCounters
    """

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
