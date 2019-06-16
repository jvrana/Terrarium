from terrarium.utils.grouper import Grouper, GroupCounter
from terrarium.utils.validate import validate_with_schema
from itertools import chain


def dict_intersection(a, b, func):
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


def group_by(a, key):
    """Groups a list by a key function"""
    d = {}
    for _a in a:
        d.setdefault(key(_a), []).append(_a)
    return d
