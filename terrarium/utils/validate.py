def validate_with_schema(data, schema, reasons=None, strict=False):
    if reasons is None:
        reasons = []
    if isinstance(schema, dict) and isinstance(data, dict):
        # schema is a dict of types or other dicts
        return all(
            k in data
            and validate_with_schema(data[k], schema[k], reasons, strict=strict)
            for k in schema
        )
    if isinstance(schema, list) and isinstance(data, list):
        # schema is list in the form [type or dict]
        return all(
            validate_with_schema(c, schema[0], reasons, strict=strict) for c in data
        )
    elif isinstance(schema, type):
        # schema is the type of conf
        return isinstance(data, schema)
    else:
        if strict:
            return type(schema) is type(data) and data == schema
        else:
            return data == schema
