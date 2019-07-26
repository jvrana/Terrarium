from os.path import dirname, abspath
import json
import pytest
from terrarium import AutoPlannerModel, NetworkFactory
from terrarium.jsoninterpreter import JSONInterpreter
from copy import deepcopy
import arrow
from pydent.planner import Planner

here = dirname(abspath(__file__))


# def make_query(sess, query_json):
#     interface = getattr(sess, query_json["model_class"])
#     method = query_json.get("method", "where")
#     if method == "where":
#
#         def where(query=None, **kwargs):
#             return interface.where(query, **kwargs)
#
#         f = where
#     else:
#         f = getattr(interface, method)
#     args = query_json.get("args", tuple())
#     kwargs = query_json.get("kwargs", {})
#     if "query" in query_json:
#         if query_json["query"]:
#             kwargs["query"] = query_json["query"]
#             if isinstance(kwargs["query"], list):
#                 models = []
#                 for q in kwargs["query"]:
#                     _kwargs = deepcopy(kwargs)
#                     _kwargs["query"] = q
#                     models += f(*args, **_kwargs)
#                 return models
#         else:
#             return []
#     return f(*args, **kwargs)
#
#
# def test_file_parser(session):
#     q = {"model_class": "Plan", "method": "last", "args": [30], "query": None}
#
#     with session.with_cache(timeout=60) as sess:
#         models = make_query(sess, q)
#         assert models
#         assert len(models) == 30
#
#
# def test_file_parser2(session):
#     q = {"model_class": "ObjectType", "query": {"name": "Yeast Glycerol Stock"}}
#
#     object_types = make_query(session, q)
#     print(object_types)
#
#
# TODO: add session information
@pytest.mark.parametrize("file", ["example1.json"])
def test_parse_json(file, session):

    interpreter = JSONInterpreter(session)
    with open(file, "r") as f:
        input_json = json.load(f)
    interpreter.parse(input_json)
    interpreter.submit()
