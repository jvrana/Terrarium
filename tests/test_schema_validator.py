from terrarium.utils.validate import (
    validate_with_schema,
    validate_with_schema_errors,
    is_any_type_of,
    is_in,
)


def test_nested__schema():

    data = {"key": {"key2": "hello"}}

    schema = {"key": {"key2": int}}

    errors = validate_with_schema_errors(data, schema)
    print(errors)


def test_valid_int_schema():
    data = {"key": 1}

    true_schema = {"key": int}
    false_schema = {"key": str}

    assert validate_with_schema(data, true_schema)
    assert not validate_with_schema(data, false_schema)


def test_valid_extra_key():
    data = {"key": 1}

    true_schema = {"key": int}
    false_schema = {"key": int, "extra": str}

    assert validate_with_schema(data, true_schema)
    assert not validate_with_schema(data, false_schema)


def test_valid_list_of_int_schema():
    data = {"key": [1]}

    true_schema = {"key": [int]}
    false_schema = {"key": int}

    assert validate_with_schema(data, true_schema)
    assert not validate_with_schema(data, false_schema)


def test_validate_nested():
    data = {"key": {"name": "myname"}}

    true_schema = {"key": {"name": str}}
    false_schema = {"key": {"name": int}}

    assert validate_with_schema(data, true_schema)
    assert not validate_with_schema(data, false_schema)


def test_validate_list_of_dict():
    data = {"key": [{"name": "anyong"}]}

    true_schema = {"key": [{"name": str}]}
    false_schema = {"key": [{"name": int}]}
    assert validate_with_schema(data, true_schema)
    assert not validate_with_schema(data, false_schema)


def test_validate_floats():
    data = {"key": 1.0}

    true_schema = {"key": float}
    false_schema = {"key": int}
    assert validate_with_schema(data, true_schema)
    assert not validate_with_schema(data, false_schema)


def test_validate_exact_value_str():
    data = {"key": "hello"}

    true_schema = {"key": "hello"}
    false_schema = {"key": "nhello"}
    assert validate_with_schema(data, true_schema)
    assert not validate_with_schema(data, false_schema)


def test_validate_exact_value_float_strict():

    data = {"key": 1.0}

    true_schema = {"key": 1}
    false_schema = {"key": int}
    assert not validate_with_schema(data, true_schema, strict=True)
    assert validate_with_schema(data, true_schema, strict=False)
    assert not validate_with_schema(data, false_schema)


def test_callable_schema():

    data = {"key": 11}

    true_schema = {"key": lambda x: x % 2 == 1}
    false_schema = {"key": lambda x: x % 2 == 0}
    assert validate_with_schema(data, true_schema)
    assert not validate_with_schema(data, false_schema)


def test_callable_schema():

    data1 = {"key": None}
    data2 = {"key": 1}

    schema1 = {"key": is_any_type_of(None, int)}
    schema2 = {"key": is_any_type_of(None, float)}

    assert validate_with_schema(data1, schema1)
    assert validate_with_schema(data2, schema1)
    assert validate_with_schema(data1, schema2)
    assert not validate_with_schema(data2, schema2)


def test_callable_schema():

    data1 = {"key": None}
    data2 = {"key": 2}

    schema1 = {"key": is_in([1, None])}

    assert validate_with_schema(data1, schema1)
    assert not validate_with_schema(data2, schema1)
