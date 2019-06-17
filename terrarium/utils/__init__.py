from terrarium.utils.grouper import Grouper, GroupCounter
from terrarium.utils.validate import validate_with_schema, validate_with_schema_errors
from itertools import chain
from more_itertools import map_reduce


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
    return dict(map_reduce(a, keyfunc=key))


def multi_group_by(d, keyfuncs, valuefunc=None, reducefunc=None):
    if callable(keyfuncs):
        keyfuncs = [keyfuncs]
    groups = map_reduce(
        d, keyfunc=keyfuncs[0], valuefunc=valuefunc, reducefunc=reducefunc
    )
    if keyfuncs[1:]:
        return {k: multi_group_by(v, keyfuncs=keyfuncs[1:]) for k, v in groups.items()}
    else:
        return groups


def multi_group_by_key(d, keys, valuefunc=None, reducefunc=None):
    return multi_group_by(
        d, [lambda x: x[k] for k in keys], valuefunc=valuefunc, reducefunc=reducefunc
    )
