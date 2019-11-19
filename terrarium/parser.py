import json
import os
from copy import deepcopy

import arrow
import networkx as nx
from pydent.planner import Planner

from terrarium import AutoPlannerModel
from terrarium import NetworkFactory
from terrarium.__version__ import __version__
from terrarium.exceptions import TerrariumJSONParseError
from terrarium.exceptions import ValidationError
from terrarium.utils.async_wrapper import make_async
from terrarium.utils.validator import Each
from terrarium.utils.validator import InstanceOf
from terrarium.utils.validator import Length
from terrarium.utils.validator import Required
from terrarium.utils.validator import validate


# TODO: implement dry run
class JSONInterpreter:

    PLAN_ID = "PLAN_ID"
    EDGES = "EDGES"
    OBJECT_TYPE = "OBJECT_TYPE"
    SAMPLE = "SAMPLE"
    GOALS = "GOALS"
    TRAIN = "TRAIN"
    MODEL_PATH = "MODEL_PATH"
    GLOBAL_CONSTRAINTS = "GLOBAL_CONSTRAINTS"
    EXCLUDE = "EXCLUDE"

    query_schema = {
        "model_class": [InstanceOf(str)],
        "method": [InstanceOf(str)],
        "queries": [InstanceOf(str)],
        "query": [InstanceOf(dict)],
        "args": [InstanceOf(list)],
    }

    schema = {
        TRAIN: [Required, query_schema],
        MODEL_PATH: [InstanceOf(str)],
        GOALS: [
            Required,
            Each(
                {
                    PLAN_ID: [InstanceOf(str)],
                    SAMPLE: [Required, query_schema],
                    OBJECT_TYPE: [query_schema],
                    EDGES: [Each([Length(2, maximum=2)])],
                }
            ),
        ],
        GLOBAL_CONSTRAINTS: [{EXCLUDE: [Each([InstanceOf(dict)])]}],
    }

    def __init__(self, session):
        self.session = session
        self.plans = {}

    @classmethod
    def validate(cls, input_json):
        result = validate(cls.schema, input_json)
        if not result[0]:
            raise ValidationError(result[1])

    def load_model(self, model_json, filename=None):
        print(os.getcwd())
        if filename and os.path.isfile(filename):
            return AutoPlannerModel.load(filename)
        else:
            if "model_class" not in model_json:
                model_json["model_class"] = "Plan"

            print("Building model...")
            model = AutoPlannerModel(
                self.session.browser, plans=self.make_query(model_json)
            )
            model.build()
            if filename:
                model.dump(filename)
            return model

    def make_query(self, query_json):
        interface = getattr(self.session, query_json["model_class"])
        method = query_json.get("method", "where")
        if method == "where":

            def where(query=None, **kwargs):
                return interface.where(query, **kwargs)

            f = where
        else:
            f = getattr(interface, method)
        args = query_json.get("args", tuple())
        kwargs = query_json.get("kwargs", {})
        if "query" in query_json:
            if query_json["query"]:
                kwargs["query"] = query_json["query"]
                if isinstance(kwargs["query"], list):
                    models = []
                    for q in kwargs["query"]:
                        _kwargs = deepcopy(kwargs)
                        _kwargs["query"] = q
                        models += f(*args, **_kwargs)
                    return models
            else:
                return []
        return f(*args, **kwargs)

    def parse(self, input_json):
        self.validate(input_json)
        utc = arrow.utcnow()
        timestamp = utc.timestamp

        with self.session.with_cache(timeout=60) as sess:
            model = self.load_model(
                input_json["TRAIN"], input_json.get("MODEL_PATH", None)
            )
            for exclude in input_json["GLOBAL_CONSTRAINTS"]["EXCLUDE"]:
                if exclude["model_class"] == "OperationType":
                    ignore_ots = self.make_query(exclude)
                    assert ignore_ots
                    model.exclude_operation_types(ignore_ots)
                elif exclude["model_class"] == "Item":
                    ignore_items = self.make_query(exclude)
                    ignore = [item.id for item in ignore_items]
                    model.add_model_filter(
                        "Item",
                        AutoPlannerModel.EXCLUDE_FILTER,
                        lambda m: m.id in ignore,
                    )
                else:
                    raise ValueError(
                        "Model class '{}' not supported for global constraints query".format(
                            exclude["model_class"]
                        )
                    )

            factory = NetworkFactory(model)

            goals = []
            for goal_num, goal in enumerate(input_json["GOALS"]):
                if "SAMPLE" not in goal:
                    raise ValueError(
                        "Key '{}' missing from input json.".format("SAMPLE")
                    )
                if "OBJECT_TYPE" not in goal:
                    raise ValueError(
                        "Key '{}' missing from input json.".format("OBJECT_TYPE")
                    )
                object_types = self.make_query(goal["OBJECT_TYPE"])
                if not object_types:
                    raise ValueError(
                        "Could not find object type with query\n{}".format(
                            json.dumps(goal["OBJECT_TYPE"], indent=2)
                        )
                    )
                samples = self.make_query(goal["SAMPLE"])
                if not samples:
                    raise ValueError(
                        "Could not find sample with query\n{}".format(
                            json.dumps(goal["SAMPLE"], indent=2)
                        )
                    )
                goal_sample = samples[0]

                goals.append(
                    {
                        "object_type": object_types[0],
                        "edges": goal.get("EDGES", []),
                        "sample": goal_sample,
                        "plan_id": goal.get(
                            "PLAN_ID", "Design{num}".format(num=goal_num)
                        )
                        + "__{}_(Terrarium v{})".format(str(timestamp), __version__),
                    }
                )

            self.plans = {}

            @make_async(4)
            def submit_goals(goals):
                for goal_num, goal in enumerate(goals):
                    if goal["plan_id"] not in self.plans:
                        canvas = Planner(sess)
                        canvas.name = goal["plan_id"]
                        canvas.plan.operations = []
                        self.plans[goal["plan_id"]] = canvas
                    else:
                        canvas = self.plans[goal["plan_id"]]

                    scgraph = nx.DiGraph()

                    for n1, n2 in goal["edges"]:
                        s1 = model.browser.find_by_name(n1)
                        s2 = model.browser.find_by_name(n2)
                        scgraph.add_node(s1.id, sample=s1)
                        scgraph.add_node(s2.id, sample=s2)
                        scgraph.add_edge(s1.id, s2.id)
                    scgraph.add_node(goal["sample"].id, sample=goal["sample"])

                    network = factory.new_from_composition(scgraph)
                    network.update_sample_composition()
                    network.run(goal["object_type"])
                    network.plan(canvas=canvas)

            submit_goals(goals)

    def submit(self):
        for plan_id, plan in self.plans.items():
            if plan.plan.operations:
                plan.prettify()
            plan.save()
        return {k: v for k, v in self.plans.items()}
