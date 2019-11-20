import json
from os.path import abspath
from os.path import dirname
from os.path import join

import pytest

from terrarium.parser import JSONInterpreter

here = dirname(abspath(__file__))


@pytest.mark.parametrize("file", ["example2.json"])
def test_parse_json(file):

    with open(join(here, file), "r") as f:
        JSONInterpreter.validate(json.load(f))
