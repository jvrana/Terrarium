from terrarium.utils.grouper import Grouper, GroupCounter
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


def validate_with_schema(conf, struct, reasons=None):
    if reasons is None:
        reasons = []
    if isinstance(struct, dict) and isinstance(conf, dict):
        # struct is a dict of types or other dicts
        return all(
            k in conf and validate_with_schema(conf[k], struct[k], reasons)
            for k in struct
        )
    if isinstance(struct, list) and isinstance(conf, list):
        # struct is list in the form [type or dict]
        return all(validate_with_schema(c, struct[0], reasons) for c in conf)
    elif isinstance(struct, type):
        # struct is the type of conf
        return isinstance(conf, struct)
    else:
        return False
