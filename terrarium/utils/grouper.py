class Grouper(object):
    class NONETYPE(object):
        pass

    def __init__(self, data=None, accumulator=None):
        if data is None:
            data = []
        self._data = data
        self._group_functions = {}
        self._groups = {}
        self._accumulator = accumulator

    @property
    def data(self):
        return tuple(self._data)

    @data.setter
    def data(self, d):
        self.clear()
        self.update(d)

    def clear(self):
        for k in self._groups:
            self._groups[k] = {}

    def reset(self):
        self.clear()
        self.update_groups(self.data)

    def update_groups(self, data):
        for name, func in self._group_functions.items():
            for d in data:
                key = func(d)
                if self._accumulator is None:
                    self._groups[name].setdefault(key, []).append(d)
                else:
                    self._accumulator(self._groups[name], key, d)

    def update(self, data):
        self._data += data
        self.update_groups(data)

    def group(self, name, function):
        self._group_functions[name] = function
        self._groups[name] = {}
        self.update(self.data)

    def get(self, name, item, default=NONETYPE):
        key = self._group_functions[name](item)
        if default is not self.NONETYPE:
            return self.to_dict()[name].get(key, default)
        else:
            return self.to_dict()[name][key]

    def getdefault(self, name, item, default=None):
        key = self._group_functions[name](item)
        return self.to_dict()[name].get(item, default)

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        grouper = Grouper()
        grouper._group_functions = self._group_functions
        grouper._data = list(self._data)
        grouper._groups = self.to_dict()
        grouper._accumulator = self._accumulator
        return grouper

    def _group_copy(self, group):
        return group[:]

    def to_dict(self):
        copy = {}
        for gname, group in self._groups.items():
            group_copy = {}
            for k, v in group.items():
                group_copy[k] = self._group_copy(v)
            copy[gname] = group_copy
        return copy

    def __getitem__(self, key):
        return self._groups[key]


class GroupCounter(Grouper):
    def __init__(self, data=None):
        def counter(group, key, item):
            if key not in group:
                group[key] = 1
            else:
                group[key] += 1

        super().__init__(data, accumulator=counter)

    def _group_copy(self, num):
        return num
