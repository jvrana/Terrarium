from collections import Counter


class HashCounter(object):
    """
    HashCounter counts objects according to a custom hashing function.
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

    def __setitem__(self, k, v):
        self.counter[self.hash_function(k)] = v

    def __getitem__(self, k):
        return self.counter[self.hash_function(k)]


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
