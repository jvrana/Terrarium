def is_any_type_of(*types):
    def f(x):
        if x is None and None in types:
            return True
        return any(isinstance(x, t) for t in types if t is not None)

    return f


def is_in(instances):
    return lambda x: x in instances


def validation_errors(data, schema, strict=False, keys=None, fail_fast=True):
    errors = []
    if keys is None:
        keys = []

    key_msg = lambda k: "".join(["['{}']".format(_) for _ in k])

    if isinstance(schema, dict) and isinstance(data, dict):
        # schema is a dict of types or other dicts
        for k in schema:
            if k not in data:
                errors.append("{} is not in data".format(key_msg(keys + [k])))
            else:
                new_errors = validation_errors(
                    data[k], schema[k], strict=strict, keys=keys + [k]
                )
                errors += new_errors
                if new_errors and fail_fast:
                    break
    elif isinstance(schema, list) and isinstance(data, list):
        # schema is list in the form [type or dict]
        for c in data:
            new_errors = validation_errors(c, schema[0], strict=strict, keys=keys[:])
            errors += new_errors
            if new_errors and fail_fast:
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
                "{}={} did not pass callable schema {}".format(
                    key_msg(keys), data, schema
                )
            )
    else:
        if not data == schema:
            errors.append(
                "{}={} did not match schema {}".format(key_msg(keys), data, schema)
            )
        elif strict and type(schema) is not type(data):
            errors.append(
                "{}={} did not match expected type when strict=True. {} != {}".format(
                    key_msg(keys), data, type(data), type(schema)
                )
            )
    return errors


def validate_with_schema(data, schema, strict=False):
    errors = validation_errors(data, schema, strict=strict, fail_fast=True)
    return len(errors) == 0


def validate_with_schema_errors(data, schema, strict=False, verbose=False):
    if verbose:
        errors = validation_errors(data, schema, strict=strict, fail_fast=True)
    else:
        errors = validation_errors(data, schema, strict=strict, fail_fast=False)
    return (len(errors) == 0, errors)
