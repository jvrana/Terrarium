from copy import deepcopy
from functools import reduce
from uuid import uuid4

import dill
import networkx as nx
from pydent.browser import Browser
from pydent.utils import filter_list
from pydent.utils.logger import Loggable
from tqdm import tqdm
from collections import Counter


class EdgeWeightContainer(Loggable):
    DEFAULT_DEPTH = 100
    DEFAULT = "DEFAULT"

    def __init__(self, browser, edge_hash, node_hash, depth=None,
                 plans=None, cost_function=None):
        """
        EdgeCalculator initializer

        :param browser: the Browser object
        :type browser: Browser
        :param edge_hash: The edge hashing function. Should take exactly 2 arguments.
        :type edge_hash: function
        :param node_hash: The node hashing function. Should take exectly 1 argument.
        :type node_hash: function
        :param plans: optional list of plans
        :type plans: list
        :param depth: the number of plans to use in the caching. If None, will default either to the length of
        the plans provided or, if that is not provided, to self.DEFAULT_DEPTH (100)
        :type depth: int
        """

        self.init_logger("EdgeCalculator@{}".format(browser.session.url))
        self.browser = browser

        if cost_function is None:
            self._cost_function = self.default_cost_function
        else:
            self._cost_function = cost_function

        self._plans = plans
        self._depth = depth

        if not plans:
            if not depth:
                self.depth = self.DEFAULT_DEPTH
            self.plans = self.browser.last(self.depth, "Plan")
        else:
            self.depth = len(plans)

        self._edge_hash = edge_hash
        self._node_hash = node_hash
        self._edge_counter = Counter()
        self._node_counter = Counter()
        self._weights = {}
        self.is_cached = False

    @property
    def plans(self):
        return list(self._plans)[-self.depth:]

    @plans.setter
    def plans(self, plans):
        """Sets the plans. Automatically """
        self._plans = plans
        self.is_cached = False

    @property
    def depth(self):
        return self._depth

    @depth.setter
    def depth(self, val):
        self._depth = val
        self.is_cached = False

    def reset(self):
        self.is_cached = False
        self._edge_counter = Counter()
        self._node_counter = Counter()

    def recompute(self):
        """Reset the counters and recompute weights"""
        self._edge_counter = Counter()
        self._node_counter = Counter()
        return self.compute()

    def compute(self):
        """Compute the weights. If previously computed, this function will avoid re-caching plans and wires."""
        if not self.is_cached:
            self.cache_plans(self.plans)
        wires = self.collect_wires(self.plans)
        operations = self.collect_operations(self.plans)
        if not self.is_cached:
            self.cache_wires(wires)
        self.is_cached = True

        edges = self.to_edges(wires, operations)
        self.update_tally(edges)
        self.save_weights(edges)

    @staticmethod
    def collect_wires(plans):
        for p in plans:
            p.wires
        all_wires = reduce((lambda x, y: x + y), [p.wires for p in plans])
        return all_wires

    @staticmethod
    def collect_operations(plans):
        all_operations = reduce((lambda x, y: x + y), [p.operations for p in plans])
        return all_operations

    def cache_wires(self, wires):
        self.browser.recursive_retrieve(wires[:], {
            "source": {
                "field_type": [],
                "operation": "operation_type",
                "allowable_field_type": []
            },
            "destination": {
                "field_type": [],
                "operation": "operation_type",
                "allowable_field_type": []
            }
        }, strict=False)

    def cache_plans(self, plans):
        self.browser.recursive_retrieve(plans, {
            "operations": {"field_values": ["allowable_field_type", "wires_as_source", "wires_as_dest"]}},
                                        strict=False)

    @staticmethod
    def to_edges(wires, operations):
        """Wires and operations to a list of edges"""
        edges = []
        for wire in wires:  # external wires
            if wire.source and wire.destination:
                edges.append((wire.source.allowable_field_type, wire.destination.allowable_field_type))
        for op in operations:  # internal wires
            for i in op.inputs:
                for o in op.outputs:
                    edges.append((i.allowable_field_type, o.allowable_field_type))
        return edges

    def update_tally(self, edges):
        for n1, n2 in tqdm(edges, desc="counting edges"):
            if n1 and n2:
                self._edge_counter[self._edge_hash(n1, n2)] += 1
                self._node_counter[self._node_hash(n1)] += 1

    def get_edge_count(self, n1, n2):
        return self._edge_counter[self._edge_hash(n1, n2)]

    def get_node_count(self, n):
        return self._node_counter[self._node_hash(n)]

    def save_weights(self, edges):
        for n1, n2 in edges:
            if n1 and n2:
                self._weights[self._edge_hash(n1, n2)] = self.cost(n1, n2)

    def cost(self, n1, n2):
        return self._cost_function(n1, n2)

    def default_cost_function(self, n1, n2):
        p = 1e-4

        n = self.get_edge_count(n1, n2) * 1.0
        t = self.get_node_count(n1) * 1.0

        if t > 0:
            p = n / t
        w = (1 - p) / (1 + p)
        return 10 / (1.0001 - w)

    def get_weight(self, n1, n2):
        if not self.is_cached:
            raise Exception("The tally and weights have not been computed")
        ehash = self._edge_hash(n1, n2)
        return self._weights.get(ehash, self.cost(n1, n2))


class AutoPlanner(Loggable):

    def __init__(self, session):
        self.browser = Browser(session)
        self.weight_container = EdgeWeightContainer(self.browser, self.hash_afts, self.external_aft_hash)
        self.template_graph = None
        self.init_logger("AutoPlanner@{url}".format(url=session.url))

    def set_verbose(self, verbose):
        if self.weight_container:
            self.weight_container.set_verbose(verbose)
        super().set_verbose(verbose)

    @staticmethod
    def external_aft_hash(aft):
        """A has function representing two 'extenal' :class:`pydent.models.AllowableFieldType` models (i.e. a wire)"""
        if not aft.field_type:
            return str(uuid4())
        if aft.field_type.part:
            part = True
        else:
            part = False
        return "{object_type}-{sample_type}-{part}".format(
            object_type=aft.object_type_id,
            sample_type=aft.sample_type_id,
            part=part,
        )

    @staticmethod
    def internal_aft_hash(aft):
        """A has function representing two 'internal' :class:`pydent.models.AllowableFieldType` models (i.e. an operation)"""
        return "{operation_type}".format(
            operation_type=aft.field_type.parent_id,
            routing=aft.field_type.routing,
            sample_type=aft.sample_type_id
        )

    @classmethod
    def hash_afts(cls, aft1, aft2):
        """Make a unique hash for a :class:`pydent.models.AllowableFieldType` pair"""
        source_hash = cls.external_aft_hash(aft1)
        dest_hash = cls.external_aft_hash(aft2)
        return "{}->{}".format(source_hash, dest_hash)

    def cache_afts(self):
        """Cache :class:`AllowableFieldType`"""
        ots = self.browser.where({"deployed": True}, "OperationType")

        self._info("Caching all AllowableFieldTypes from {} deployed operation types".format(len(ots)))

        results = self.browser.recursive_retrieve(ots, {
            "field_types": {
                "allowable_field_types": {
                    "object_type": [],
                    "sample_type": [],
                    "field_type": []
                }
            }
        }, strict=False
                                                  )
        fts = [ft for ft in results['field_types'] if ft.ftype == 'sample']
        inputs = [ft for ft in fts if ft.role == 'input']
        outputs = [ft for ft in fts if ft.role == 'output']

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

    def construct_graph_edges(self):
        """
        Construct edges from all deployed allowable_field_types

        :return: list of tuples representing connections between AllowableFieldType
        :rtype: list
        """
        input_afts, output_afts = self.cache_afts()

        external_groups = {}
        for aft in input_afts:
            external_groups.setdefault(self.external_aft_hash(aft), []).append(aft)

        internal_groups = {}
        for aft in output_afts:
            internal_groups.setdefault(self.internal_aft_hash(aft), []).append(aft)

        edges = []
        for oaft in tqdm(output_afts, desc="hashing output AllowableFieldTypes"):
            hsh = self.external_aft_hash(oaft)
            externals = external_groups.get(hsh, [])
            for aft in externals:
                edges.append((oaft, aft))

        for iaft in tqdm(input_afts, desc="hashing input AllowableFieldTypes"):
            hsh = self.internal_aft_hash(iaft)
            internals = internal_groups.get(hsh, [])
            for aft in internals:
                edges.append((iaft, aft))
        return edges

    def construct_template_graph(self, ignore=()):
        """
        Construct a graph of all possible Operation connections.

        :param depth:
        :type depth:
        :return:
        :rtype:
        """
        # computer weights
        self.weight_container.compute()

        G = nx.DiGraph()
        edges = self.construct_graph_edges()
        for n1, n2 in edges:
            G.add_node(n1.id, model_class=n1.__class__.__name__, model=n1)
            G.add_node(n2.id, model_class=n2.__class__.__name__, model=n2)
            if n1 and n2:
                G.add_edge(n1.id, n2.id, weight=self.weight_container.get_weight(n1, n2))

        self._info("Building Graph:")
        self._info("  {} edges".format(len(list(G.edges))))
        self._info("  {} nodes".format(len(G)))

        all_afts = [self.browser.find(n, "AllowableFieldType") for n in G.nodes()]

        # filter by operation type category
        ignore_ots = self.browser.where({"category": "Control Blocks"}, "OperationType")
        nodes = [aft.id for aft in all_afts if
                 aft.field_type.parent_id and aft.field_type.parent_id not in [ot.id for ot in ignore_ots]]
        template_graph = G.subgraph(nodes)
        self._info("Graph size reduced from {} to {} nodes".format(len(G), len(template_graph)))

        print("Example edges")
        for e in list(template_graph.edges)[:3]:
            n1 = self.browser.find(e[0], "AllowableFieldType")
            n2 = self.browser.find(e[1], "AllowableFieldType")
            print()
            print("{} {}".format(n1.field_type.role, n1))
            print("{} {}".format(n2.field_type.role, n2))

        print("Example edges")
        for edge in list(template_graph.edges)[:5]:
            print(template_graph[edge[0]][edge[1]])
        self.template_graph = template_graph
        return template_graph

    def copy_graph(self):
        """Returns a copy of the template graph, along with its data."""
        graph = nx.DiGraph()
        graph.add_nodes_from(self.template_graph.nodes(data=True))
        graph.add_edges_from(self.template_graph.edges(data=True))
        return graph

    def collect_afts(self, graph):
        """
        Collect :class:`pydent.models.AllowableFieldType` models from graph

        :param graph:
        :type graph:
        :return:
        :rtype:
        """
        input_afts = []
        output_afts = []

        for n in graph.nodes:
            node = graph.node[n]
            if node['model_class'] == "AllowableFieldType":
                if node['model'].field_type.role == "input":
                    input_afts.append(node['model'])
                elif node['model'].field_type.role == 'output':
                    output_afts.append(node['model'])
        return input_afts, output_afts

    def print_path(self, path, graph):
        ots = []
        for aftid in path:
            aft = self.browser.find(aftid, 'AllowableFieldType')
            if aft:
                ot = self.browser.find(aft.field_type.parent_id, 'OperationType')
                ots.append("{ot}".format(role=aft.field_type.role, name=aft.field_type.name, ot=ot.name))

        edge_weights = [graph[x][y]['weight'] for x, y in zip(path[:-1], path[1:])]
        print("PATH: {}".format(path))
        print('WEIGHTS: {}'.format(edge_weights))
        print("NUM NODES: {}".format(len(path)))
        print("OP TYPES:\n{}".format(ots))

    def search_graph(self, goal_sample, goal_object_type, start_object_type):
        graph = self.copy_graph()

        # filter afts
        input_afts, output_afts = self.collect_afts(graph)
        obj1 = start_object_type
        obj2 = goal_object_type
        afts1 = filter_list(input_afts, object_type_id=obj1.id, sample_type_id=obj1.sample_type_id)
        afts2 = filter_list(output_afts, object_type_id=obj2.id, sample_type_id=obj2.sample_type_id)

        # Add terminal nodes
        graph.add_node("START")
        graph.add_node("END")
        for aft in afts1:
            graph.add_edge("START", aft.id, weight=0)
        for aft in afts2:
            graph.add_edge(aft.id, "END", weight=0)

        # find and sort shortest paths
        shortest_paths = []
        for n1 in graph.successors("START"):
            n2 = "END"
            try:
                path = nx.dijkstra_path(graph, n1, n2, weight='weight')
                path_length = nx.dijkstra_path_length(graph, n1, n2, weight='weight')
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

    def dump(self, path):
        with open(path, 'wb') as f:
            dill.dump(self, f)

    def load(self, path):
        with open(path, 'rb') as f:
            dill.load(f)
