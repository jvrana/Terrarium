from .grouper import Grouper, GroupCounter
from itertools import chain
from more_itertools import map_reduce
from typing import Callable, List, Any
from .test_utils import timeit
from .logger import logger


def dict_intersection(a: dict, b: dict, func: Callable) -> dict:
    """
    Returns a new dictionary with `func` applied
    to values of `func(a[key], b[key])` where `key` is a
    key shared between dictionaries `a` and `b`

    :param a:
    :param b:
    :param func:
    :return:
    """
    c = {}
    for k in chain(a, b):
        if k in a and k in b:
            c.setdefault(k, []).append(func(a[k], b[k]))
    return c


def group_by(a: List[Any], key: Callable) -> dict:
    """Groups a list by a key function"""
    return dict(map_reduce(a, keyfunc=key))


def multi_group_by(d, keyfuncs: List[Callable], valuefunc=None, reducefunc=None):
    """
    Returns a nested dictionary keyed by functions found in the `keyfuncs`.
    For example, the following would create a nested dictionary of values that are (i) odd
    and (ii) divisible by 3.

    .. code-block:

        multi_group_by([1,2,3,4], keyfuncs=[lambda x: x%2 == 0, lambda x: x%3 == 0])

    :param d: data dict
    :param keyfuncs: list of key functions to
    :param valuefunc: function to convert values
    :param reducefunc: reduce function
    :return:
    """
    if callable(keyfuncs):
        keyfuncs = [keyfuncs]
    groups = map_reduce(
        d, keyfunc=keyfuncs[0], valuefunc=valuefunc, reducefunc=reducefunc
    )
    if keyfuncs[1:]:
        return {k: multi_group_by(v, keyfuncs=keyfuncs[1:]) for k, v in groups.items()}
    else:
        return groups


def multi_group_by_key(d, keys: List[Any], valuefunc=None, reducefunc=None):
    """
    Returns a nested dictionary keyed by a key.

    :param d: data dict
    :param keys: list of keys
    :param valuefunc: function to convert values
    :param reducefunc: reduce function
    :return:
    """
    return multi_group_by(
        d, [lambda x: x[k] for k in keys], valuefunc=valuefunc, reducefunc=reducefunc
    )


def flatten_json(data: dict, sep=".") -> dict:
    """
    Converts a nested dictionary to a flattened dictionary with keys squashed by a separator

    :param data: the input dictionaruy
    :param sep: squashing seperatory
    :return:
    """
    out = {}

    def flatten(x, name=""):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + sep)
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + sep)
                i += 1
        else:
            out[name[:-1]] = x

    flatten(data)
    return out
