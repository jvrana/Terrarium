from os.path import dirname, abspath
import pytest
import json
from terrarium.parser import JSONInterpreter

here = dirname(abspath(__file__))


@pytest.mark.parametrize("file", ["example1.json"])
def test_parse_json(file):

    with open(file, "r") as f:
        JSONInterpreter.validate(json.load(f))
