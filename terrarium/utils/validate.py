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
