from terrarium.utils import validate_with_schema


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
