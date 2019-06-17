import inspect
from functools import wraps


def is_any_type_of(*types):
    @wraps(is_any_type_of)
    def f(x, types=types):
        if x is None and None in types:
            return True
        return any(isinstance(x, t) for t in types if t is not None)

    return f


def is_in(instances):
    @wraps(is_in)
    def is_in_wrapper(x, instances=instances):
        return x in instances

    return is_in_wrapper


def validation_errors(
    data, schema, strict=False, keys=None, fail_fast=True, value_comparator=None
):
    errors = []
    if keys is None:
        keys = []

    def key_msg(keys):
        msg = ""
        for key in keys:
            if isinstance(key, str):
                msg += "['{}']".format(key)
            else:
                msg += "[{}]".format(key)
        return msg

    if isinstance(schema, dict) and isinstance(data, dict):
        # schema is a dict of types or other dicts
        for k in schema:
            if k not in data:
                errors.append("{} is not in data".format(key_msg(keys + [k])))
            else:
                new_errors = validation_errors(
                    data[k],
                    schema[k],
                    strict=strict,
                    keys=keys + [k],
                    value_comparator=value_comparator,
                )
                errors += new_errors
                if new_errors and fail_fast:
                    break
    elif isinstance(schema, list) and isinstance(data, list):
        # schema is list in the form [type or dict]
        for i, c in enumerate(data):
            if all(
                validation_errors(
                    c,
                    s,
                    strict=strict,
                    keys=keys + [i],
                    value_comparator=value_comparator,
                )
                for s in schema
            ):
                errors.append(
                    "{}={} did not match any schemas in {}".format(
                        key_msg(keys + [i]), c, schema
                    )
                )
                if fail_fast:
                    break
    elif isinstance(schema, type):
        # schema is the type of conf
        if not isinstance(data, schema):
            errors.append(
                "{}={} is not an instance of {}".format(key_msg(keys), data, schema)
            )
    elif callable(schema):
        if not schema(data):
            errors.append(
                "{}={} did not pass callable schema value {} from {} {}".format(
                    key_msg(keys),
                    data,
                    schema(data),
                    schema,
                    inspect.getfullargspec(schema),
                )
            )
    else:
        if strict and type(schema) is not type(data):
            errors.append(
                "{}={} did not match expected type when strict=True. {} != {}".format(
                    key_msg(keys), data, type(data), type(schema)
                )
            )
        elif value_comparator:
            if not value_comparator(data, schema):
                errors.append(
                    "{}={} did not match schema {} using {}".format(
                        key_msg(keys), data, schema, value_comparator
                    )
                )
        elif data != schema:
            errors.append(
                "{}={} did not equal schema {}".format(key_msg(keys), data, schema)
            )
    return errors


def validate_with_schema(data, schema, strict=False, value_comparator=None):
    errors = validation_errors(
        data, schema, strict=strict, fail_fast=True, value_comparator=value_comparator
    )
    return len(errors) == 0


def validate_with_schema_errors(
    data, schema, strict=False, verbose=False, value_comparator=None
):
    if verbose:
        errors = validation_errors(
            data,
            schema,
            strict=strict,
            fail_fast=True,
            value_comparator=value_comparator,
        )
    else:
        errors = validation_errors(
            data,
            schema,
            strict=strict,
            fail_fast=False,
            value_comparator=value_comparator,
        )
    return len(errors) == 0, errors
