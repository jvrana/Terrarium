from terrarium import AutoPlannerModel, NetworkFactory
from pydent.planner import Planner
import json
import arrow
from copy import deepcopy
import os


class JSONInterpreter(object):
    def __init__(self, session):
        self.session = session
        self.plans = {}

    def load_model(self, model_json):
        if "filename" in model_json:
            if os.path.isfile(model_json["filename"]):
                return AutoPlannerModel.load(model_json["filename"])

        if "model_class" not in model_json:
            model_json["model_class"] = "Plan"

        print("Building model...")
        model = AutoPlannerModel(
            self.session.browser, plans=self.make_query(model_json)
        )
        model.build()
        if "filename" in model_json:
            model.dump(model_json["filename"])
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
        utc = arrow.utcnow()
        timestamp = utc.timestamp

        with self.session.with_cache(timeout=60) as sess:
            model = self.load_model(input_json["TRAIN"])
            for exclude in input_json["GLOBAL_CONSTRAINTS"]["EXCLUDE"]:
                if exclude["model_class"] == "OperationType":
                    ignore_ots = self.make_query(exclude)
                    ignore = [ot.id for ot in ignore_ots]
                    model.add_model_filter(
                        "AllowableFieldType",
                        AutoPlannerModel.EXCLUDE_FILTER,
                        lambda m: m.field_type.parent_id in ignore,
                    )
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
                            "PLAN_ID",
                            "Design{num}".format(num=goal_num) + "_" + str(timestamp),
                        ),
                    }
                )

            plans = {}

            for goal_num, goal in enumerate(goals):
                plans.setdefault(goal["plan_id"], Planner(sess))
                plan = plans[goal["plan_id"]]
                plan.name = goal["plan_id"]
                scgraph = factory.sample_composition_from_edges(goal.get("EDGES", []))
                scgraph.add_node(goal["sample"].id, sample=goal["sample"])
                network = factory.new_from_composition(scgraph)
                network.run(goal["object_type"])
                network.plan(canvas=plan)

    def submit(self):
        for plan_id, plan in self.plans.items():
            plan.save()
        return {k: v.plan.id for k, v in self.plans.items()}
