from os.path import dirname, abspath, join
import json
import pytest
from terrarium.parser import JSONInterpreter

here = dirname(abspath(__file__))


@pytest.mark.parametrize("file", ["example1.json"])
def test_parse_json(file, session):

    with session.with_cache(timeout=60) as sess:
        interpreter = JSONInterpreter(sess)
    with open(join(here, file), "r") as f:
        input_json = json.load(f)
        input_json["MODEL_PATH"] = join(here, input_json["MODEL_PATH"])
        interpreter.parse(input_json)
        plans = interpreter.submit()
        print(plans)
