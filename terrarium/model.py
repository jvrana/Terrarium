import json
import sys
import warnings
from functools import reduce
from os import stat
from uuid import uuid4

import arrow
import dill
import networkx as nx
from pydent.browser import Browser
from pydent.utils.logger import Loggable
from pydent.models import User
from tqdm import tqdm

from terrarium.__version__ import __version__
from terrarium.exceptions import AutoPlannerException, AutoPlannerLoadingError
from terrarium.utils.hash_utils import HashCounter
from terrarium.browser_graph import BrowserGraph

from functools import wraps


class SetRecusion(object):

    DEFAULT = 2000

    def __init__(self, limit, default=2000):
        self.limit = limit
        self.default = default

    def __enter__(self):
        sys.setrecursionlimit(self.limit)

    def __exit__(self, exception_type, exception_value, traceback):
        sys.setrecursionlimit(self.default)

    @classmethod
    def set_recursion_limit(cls, limit, default=DEFAULT):
        def wrapper(fxn):
            @wraps(fxn)
            def _wrapped(*args, **kwargs):
                with cls(limit, default):
                    result = fxn(*args, **kwargs)
                return result

            return _wrapped

        return wrapper


class EdgeWeightContainer(Loggable):
    def __init__(self, browser, edge_hash, node_hash, plans, plan_ids=None):
        """
        EdgeCalculator initializer

        :param browser: the Browser object
        :type browser: Browser
        :param edge_hash: The edge hashing function. Should take exactly 2 arguments.
        :type edge_hash: function
        :param node_hash: The node hashing function. Should take exactly 1 argument.
        :type node_hash: function
        :param plans: optional list of plans
        :type plans: list
        """
        self.browser = browser
        self.init_logger("EdgeWeightContainer({})".format(self.browser.session))
        # filter only those plans that have operations
        self._plans = []
        if plan_ids is not None:
            self._plan_ids = plan_ids
        else:
            self._plan_ids = []
        if plans is not None:
            self.plans = plans

        def new_edge_hash(pair):
            h = edge_hash(pair)
            return "{}_{}_{}".format(
                pair[0].field_type.parent_id, h, pair[1].field_type.parent_id
            )

        self.edges = []
        self._edge_counter = HashCounter(new_edge_hash)
        self._node_counter = HashCounter(node_hash)
        self._weights = {}
        self.created_at = str(arrow.now())
        self.updated_at = None
        self.is_cached = False

    @property
    def plans(self):
        return self._plans

    @plans.setter
    def plans(self, new_plans):
        self._plans = new_plans
        self._plan_ids = [p.id for p in new_plans]

    def _was_updated(self):
        self.updated_at = str(arrow.now())

    def reset(self):
        self.is_cached = False
        self._edge_counter.clear()
        self._node_counter.clear()

    def recompute(self):
        """Reset the counters and recompute weights"""
        self._edge_counter.clear()
        self._node_counter.clear()
        return self.compute()

    def update(self, plans, only_unique=False):
        if only_unique:
            existing_plan_ids = [p.id for p in self.plans]
            plan_ids = [p.id for p in plans]
            unique_plans = set(plan_ids).difference(existing_plan_ids)
            num_ignored = len(plans) - len(unique_plans)
            plans = list(unique_plans)
            self._info("Ignoring {} existing plans".format(num_ignored))
        self._info("Updating edge counter with {} new plans".format(len(plans)))

        self.cache_plans()

        wires = self.collect_wires()
        self._info("  {} wires loaded".format(len(wires)))

        operations = self.collect_operations()
        self._info("  {} operations loaded".format(len(operations)))

        edges = self.to_edges(wires, operations)
        self.update_tally(edges)

        self.plans += plans
        self.edges += edges
        self.save_weights(self.edges)

    def compute(self):
        """Compute the weights. If previously computed, this function will avoid re-caching plans and wires."""
        self._info("Computing weights for {} plans".format(len(self.plans)))
        self._was_updated()
        if not self.is_cached:
            self.cache_plans()
        else:
            self._info("   Plans already cached. Skipping...")
        wires = self.collect_wires()
        self._info("  {} wires loaded".format(len(wires)))
        operations = self.collect_operations()
        self._info("  {} operations loaded".format(len(operations)))
        self.is_cached = True
        self.edges = self.to_edges(wires, operations)
        self.update_tally(self.edges)
        self.save_weights(self.edges)

    def cache_plans(self):
        self._info("   Caching plans...")
        self.browser.get("Plan", {"operations": {"field_values"}})

        self.browser.get("Wire", {"source", "destination"})

        self.browser.get("FieldValue", {"allowable_field_type": "field_type"})

        self.browser.get("Operation", "operation_type")

    def collect_wires(self):
        return self.browser.get("Wire")

    def collect_operations(self):
        return self.browser.get("Operation")

    @staticmethod
    def to_edges(wires, operations):
        """Wires and operations to a list of edges"""
        edges = []
        for wire in wires:  # external wires
            if wire.source and wire.destination:
                edges.append(
                    (
                        wire.source.allowable_field_type,
                        wire.destination.allowable_field_type,
                    )
                )
        for op in operations:  # internal wires
            for i in op.inputs:
                for o in op.outputs:
                    edges.append((i.allowable_field_type, o.allowable_field_type))

        # due to DB inconsitencies, some wires and operations have not AFTS
        edges = [(n1, n2) for n1, n2 in edges if n1 is not None and n2 is not None]
        return edges

    def update_tally(self, edges):
        self._info("Hashing and counting edges...")
        for n1, n2 in tqdm(edges, desc="counting edges"):
            if n1 and n2:
                self._edge_counter[(n1, n2)] += 1
                self._node_counter[n2] += 1

    def save_weights(self, edges):
        for n1, n2 in edges:
            if n1 and n2:
                self._weights[self._edge_counter.hash_function((n1, n2))] = self.cost(
                    n1, n2
                )

    def cost(self, n1, n2):
        n = self._edge_counter[(n1, n2)] * 1.0
        t = self._node_counter[n1] * 1.0
        return self.cost_function(n, t)

    def cost_function(self, n, t):
        n = max(n, 0)
        t = max(t, 0)
        p = 10e-6
        if t > 0:
            p = n / t
        w = (1 - p) / (1 + p)
        return 10 / (1.000001 - w)

    def get_weight(self, n1, n2):
        if not self.is_cached:
            raise AutoPlannerException("The tally and weights have not been computed")
        ehash = self._edge_counter.hash_function((n1, n2))
        return self._weights.get(ehash, self.cost(n1, n2))

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        new = self.__class__(self.browser, None, None, self.plans)
        new._edge_counter = self._edge_counter.copy()
        new._node_counter = self._node_counter.copy()
        new.is_cached = self.is_cached
        return new

    def __add__(self, other):
        new = self.copy()
        new._edge_counter = self._edge_counter + other._edge_counter
        new._node_counter = self._node_counter + other._node_counter
        new._was_updated()
        return new

    def __sub__(self, other):
        new = self.copy()
        new._edge_counter = self._edge_counter - other._edge_counter
        new._node_counter = self._node_counter - other._node_counter

        # make sure there are no negative values
        for k, v in new._edge_counter.counter:
            new._edge_counter.counter[k] = max(v, 0)

        for k, v in new._node_counter.counter:
            new._node_counter.counter[k] = max(v, 0)

        new._was_updated()
        return new

    def __mul__(self, num):
        new = self.copy()
        new._edge_counter = self._edge_counter * num
        new._node_counter = self._node_counter * num
        new._was_updated()
        return new

    def __getstate__(self):
        return {
            "_edge_counter": self._edge_counter,
            "_node_counter": self._node_counter,
            "plan_ids": self._plan_ids,
            "is_cached": self.is_cached,
            "_weights": self._weights,
            "updated_at": self.updated_at,
            "created_at": self.created_at,
            "edges": self.edges,
        }

    def __setstate__(self, state):

        self._edge_counter = state["_edge_counter"]
        self._node_counter = state["_node_counter"]
        self.edges = state["edges"]
        self._plans = []
        self._plan_ids = state["plan_ids"]
        self.is_cached = state["is_cached"]
        self._weights = state["_weights"]
        self.updated_at = state["updated_at"]
        self.created_at = state["created_at"]
        # self.browser = state['browser']
        self.init_logger("EdgeWeightContainer")


class ModelFactory(object):
    """
    Build Terrarium models from a shared model cache.
    """

    def __init__(self, session):
        self.browser = Browser(session)

    def new(self, plans=None):
        """

        :param plans: list of plans
        :type plans: list
        :return:
        :rtype: AutPlannerModel
        """
        return AutoPlannerModel(self.browser, plans)

    def emulate(self, login=None, user_id=None, user=None, limit=-1):
        """Emulate a particular user (or users if supplied a list)"""
        if not user:
            query = {
                k: v
                for k, v in {"login": login, "id": user_id}.items()
                if v is not None
            }
            if not query:
                raise ValueError(
                    "You must provide either a login, user_id, or user instance"
                )
            user = self.browser.one(query=query, model_class="User")
        if not issubclass(type(user), User):
            raise ValueError(
                "Expected a {} class but found a {}".format(type(User), type(user))
            )
        plans = self.browser.last(
            limit, query=dict(user_id=user.id), model_class="Plan"
        )
        return self.new(plans)

    def union(self, models):
        new_model = AutoPlannerModel(self.browser)
        for m in models:
            new_model += m
        return new_model

    @staticmethod
    def load_model(path):
        """
        Loads a new model from a filepath
        :param path:
        :type path:
        :return:
        :rtype:
        """
        return AutoPlannerModel.load(path)


class AutoPlannerModel(Loggable):
    """
    Builds a model from historical plan data.
    """

    EXCLUDE_FILTER = "exclude"
    FILTER_MODEL_CLASS = "model_class"
    FILTER_FUNCTION = "function"
    VALID_FILTERS = [EXCLUDE_FILTER]

    def __init__(self, browser, plans=None, name=None):
        self.browser = browser
        self.init_logger("AutoPlanner@{url}".format(url=self.browser.session.url))
        if plans:
            self.plans = plans
            self.browser.update_cache(plans)
        self.weight_container = EdgeWeightContainer(
            self.browser, self._hash_afts, self._external_aft_hash, plans=plans
        )
        self._template_graph = None
        self.model_filters = {}
        if name is None:
            name = "unnamed_{}".format(id(self))
        self.name = name
        self._version = __version__
        self.created_at = str(arrow.now())
        self.updated_at = str(arrow.now())

    def info(self):
        return {
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "has_template_graph": self._template_graph is not None,
            "num_plans": len(self.weight_container._plan_ids),
        }

    def print_info(self):
        print(json.dumps(self.info(), indent=2))

    # TODO: method for printing statistics and plots for the model
    def plots(self):
        raise NotImplemented()

    @property
    def version(self):
        return self._version

    def set_plans(self, plans):
        self.weight_container.plans = plans
        self._template_graph = None

    def set_verbose(self, verbose):
        super().set_verbose(verbose)

    @staticmethod
    def _external_aft_hash(aft):
        """A has function representing two 'external' :class:`pydent.models.AllowableFieldType`
        models (i.e. a wire)"""
        if not aft.field_type:
            return str(uuid4())
        if aft.field_type.part:
            part = True
        else:
            part = False
        return "{object_type}-{sample_type}-{part}".format(
            object_type=aft.object_type_id, sample_type=aft.sample_type_id, part=part
        )

    @staticmethod
    def _internal_aft_hash(aft):
        """A has function representing two 'internal' :class:`pydent.models.AllowableFieldType`
        models (i.e. an operation)"""

        return "{operation_type}".format(
            operation_type=aft.field_type.parent_id,
            routing=aft.field_type.routing,
            sample_type=aft.sample_type_id,
        )

    @classmethod
    def _hash_afts(cls, pair):
        """Make a unique hash for a :class:`pydent.models.AllowableFieldType` pair"""
        source_hash = cls._external_aft_hash(pair[0])
        dest_hash = cls._external_aft_hash(pair[1])
        return "{}->{}".format(source_hash, dest_hash)

    def _cache_afts(self):
        """Cache :class:`AllowableFieldType`"""
        ots = self.browser.where({"deployed": True}, "OperationType")

        self._info(
            "Caching all AllowableFieldTypes from {} deployed operation types".format(
                len(ots)
            )
        )

        results = self.browser.recursive_retrieve(
            ots,
            {
                "field_types": {
                    "allowable_field_types": {
                        "object_type": [],
                        "sample_type": [],
                        "field_type": [],
                    }
                }
            },
            strict=False,
        )
        fts = [ft for ft in results["field_types"] if ft.ftype == "sample"]
        inputs = [ft for ft in fts if ft.role == "input"]
        outputs = [ft for ft in fts if ft.role == "output"]

        input_afts = []
        for i in inputs:
            for aft in i.allowable_field_types:
                if aft not in input_afts:
                    input_afts.append(aft)

        output_afts = []
        for o in outputs:
            for aft in o.allowable_field_types:
                if aft not in output_afts:
                    output_afts.append(aft)

        return input_afts, output_afts

    @classmethod
    def _match_internal_afts(cls, input_afts, output_afts):
        internal_groups = {}
        for aft in output_afts:
            internal_groups.setdefault(cls._internal_aft_hash(aft), []).append(aft)

        edges = []

        for iaft in input_afts:
            hsh = cls._internal_aft_hash(iaft)
            internals = internal_groups.get(hsh, [])
            for aft in internals:
                edges.append((iaft, aft))
        return edges

    @classmethod
    def _match_external_afts(cls, input_afts, output_afts):
        external_groups = {}
        for aft in input_afts:
            external_groups.setdefault(cls._external_aft_hash(aft), []).append(aft)

        edges = []
        for oaft in output_afts:
            hsh = cls._external_aft_hash(oaft)
            externals = external_groups.get(hsh, [])
            for aft in externals:
                edges.append((oaft, aft))
        return edges

    @classmethod
    def _match_afts(cls, input_afts, output_afts):
        return cls._match_internal_afts(
            input_afts, output_afts
        ) + cls._match_external_afts(input_afts, output_afts)

    def _get_aft_pairs(self):
        """
        Construct edges from all deployed allowable_field_types

        :return: list of tuples representing connections between AllowableFieldType
        :rtype: list
        """
        input_afts, output_afts = self._cache_afts()
        return self._match_afts(input_afts, output_afts)

    @property
    def template_graph(self):
        if self._template_graph is None:
            self.construct_template_graph()
        graph = self._template_graph
        if self.EXCLUDE_FILTER in self.model_filters:
            for f in self.model_filters[self.EXCLUDE_FILTER]:
                graph = graph.filter_out_models(
                    model_class=f[self.FILTER_MODEL_CLASS], key=f[self.FILTER_FUNCTION]
                )
        return graph

    def add_model_filter(self, model_class, filter_type, func):
        if filter_type not in self.VALID_FILTERS:
            raise ValueError(
                "Filter type '{}' not recognized. Select from {}".format(
                    filter_type, self.VALID_FILTERS
                )
            )
        self.model_filters.setdefault(filter_type, []).append(
            {self.FILTER_FUNCTION: func, self.FILTER_MODEL_CLASS: model_class}
        )

    def reset_model_filters(self):
        self.model_filters = []

    def update_weights(self, graph, weight_container):
        for aft1, aft2 in self._get_aft_pairs():

            edge_type = None
            roles = [aft.field_type.role for aft in [aft1, aft2]]
            if roles == ("input", "output"):
                edge_type = "internal"
            elif roles == ("output", "input"):
                edge_type = "external"

            graph.add_edge_from_models(
                aft1,
                aft2,
                edge_type=edge_type,
                weight=weight_container.get_weight(aft1, aft2),
            )

    def build(self):
        """
        Construct a graph of all possible Operation connections.
        """
        # computer weights
        self.weight_container.compute()

        self._info("Building Graph:")
        G = BrowserGraph(self.browser)
        self.update_weights(G, self.weight_container)
        self._info("  {} edges".format(len(list(G.edges()))))
        self._info("  {} nodes".format(len(G)))

        self._template_graph = G
        self._was_updated()
        return self

    def _collect_afts(self, graph):
        """
        Collect :class:`pydent.models.AllowableFieldType` models from graph

        :param graph: a browser graph
        :type graph: BrowserGraph
        :return: list of tuples of input vs output allowable field types in the graph
        :rtype: list
        """
        afts = graph.models("AllowableFieldType")

        input_afts = [aft for aft in afts if aft.field_type.role == "input"]
        output_afts = [aft for aft in afts if aft.field_type.role == "output"]
        return input_afts, output_afts

    def print_path(self, path, graph):
        ots = []
        for n, ndata in graph.iter_model_data("AllowableFieldType", nbunch=path):
            aft = ndata["model"]
            ot = self.browser.find(aft.field_type.parent_id, "OperationType")
            ots.append("{ot} in '{category}'".format(category=ot.category, ot=ot.name))

        edge_weights = [
            graph.get_edge(x, y)["weight"] for x, y in zip(path[:-1], path[1:])
        ]
        print("PATH: {}".format(path))
        print("WEIGHTS: {}".format(edge_weights))
        print("NUM NODES: {}".format(len(path)))
        print("OP TYPES:\n{}".format(ots))

    def search_graph(self, goal_sample, goal_object_type, start_object_type):
        graph = self.template_graph.copy()

        # filter afts
        obj1 = start_object_type
        obj2 = goal_object_type

        # Add terminal nodes
        graph.add_special_node("START", "START")
        graph.add_special_node("END", "END")
        for n, ndata in graph.iter_model_data(
            "AllowableFieldType",
            object_type_id=obj1.id,
            sample_type_id=obj1.sample_type_id,
        ):
            graph.add_edge("START", n, weight=0)
        for n, ndata in graph.iter_model_data(
            "AllowableFieldType",
            object_type_id=obj2.id,
            sample_type_id=obj2.sample_type_id,
        ):
            graph.add_edge(n, "END", weight=0)

        # find and sort shortest paths
        shortest_paths = []
        for n1 in graph.graph.successors("START"):
            n2 = "END"
            try:
                path = nx.dijkstra_path(graph.graph, n1, n2, weight="weight")
                path_length = nx.dijkstra_path_length(
                    graph.graph, n1, n2, weight="weight"
                )
                shortest_paths.append((path, path_length))
            except nx.exception.NetworkXNoPath:
                pass
        shortest_paths = sorted(shortest_paths, key=lambda x: x[1])

        # print the results
        print()
        print("*" * 50)
        print("{} >> {}".format(obj1.name, obj2.name))
        print("*" * 50)
        print()
        for path, pathlen in shortest_paths[:10]:
            print(pathlen)
            self.print_path(path, graph)

    @SetRecusion.set_recursion_limit(10000)
    def dump(self, path):
        with open(path, "wb") as f:
            dill.dump(
                {
                    "browser": self.browser,
                    "template_graph": self.template_graph,
                    "version": self._version,
                    "name": self.name,
                    "created_at": self.created_at,
                    "updated_at": self.updated_at,
                    "weight_container": self.weight_container,
                },
                f,
            )
        statinfo = stat(path)
        self._info("{} bytes written to '{}'".format(statinfo.st_size, path))

    def save(self, path):
        return self.dump(path)

    @classmethod
    @SetRecusion.set_recursion_limit(10000)
    def load(cls, path):
        with open(path, "rb") as f:
            try:
                data = dill.load(f)
                if data["version"] != __version__:
                    warnings.warn(
                        "Version number of saved model ('{}') does not match current model version ('{}')".format(
                            data["version"], __version__
                        )
                    )
                browser = data["browser"]
                model = cls(browser.session)

                statinfo = stat(path)
                model._info(
                    "{} bytes loaded from '{}' to new AutoPlanner (id={})".format(
                        statinfo.st_size, path, id(model)
                    )
                )

                model.browser = browser
                model.weight_container = data["weight_container"]
                model._template_graph = data["template_graph"]
                model._version = data["version"]
                model.name = data.get("name", None)
                model.updated_at = data.get("updated_at", None)
                model.created_at = data.get("created_at", None)
                return model
            except Exception as e:
                raise e
                msg = "An error occurred while loading an {} model:\n{}".format(
                    cls.__name__, str(e)
                )
                if "version" in data and data["version"] != __version__:
                    msg = (
                        "Version notice: This may have occurred since saved model version {} "
                        "does not match current version {}".format(
                            data["version"], __version__
                        )
                    )
                raise AutoPlannerLoadingError(msg) from e

    def _was_updated(self):
        self.updated_at = str(arrow.now())

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        new = self.__class__(self.browser, None, self.name + "_copy")
        new.weight_container = self.weight_container
        if self._template_graph:
            new._template_graph = self._template_graph.copy()
        return new

    def __add__(self, other):
        new = self.copy()
        new.weight_container = self.weight_container + other.weight_container
        if self._template_graph:
            new.update_weights(self._template_graph, new.weight_container)
        return new

    def __mul__(self, num):
        new = self.copy()
        new.weight_container = self.weight_container * num
        if self._template_graph:
            new.update_weights(self._template_graph, new.weight_container)
        return new
