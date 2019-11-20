import json
from os.path import abspath
from os.path import dirname
from os.path import join

import pytest

from terrarium.parser import JSONInterpreter

here = dirname(abspath(__file__))


@pytest.mark.parametrize(
    "file", ["example2.json", "example3.json", "example4.json", "example5.json"]
)
def test_parse_json(file, session):

    with session.with_cache(timeout=60) as sess:
        interpreter = JSONInterpreter(sess)
    with open(join(here, file), "r") as f:
        input_json = json.load(f)
        input_json["MODEL_PATH"] = join(here, input_json["MODEL_PATH"])
        interpreter.parse(input_json)
        plans = interpreter.submit()
        for k, v in plans.items():
            print("http://0.0.0.0/plans?plan_id={}".format(v.plan.id))
