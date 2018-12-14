from functools import reduce
from os import stat
from uuid import uuid4

import dill
import networkx as nx
import pandas as pd
from pydent.base import ModelBase as TridentBase
from pydent.browser import Browser
from pydent.utils.logger import Loggable
from tqdm import tqdm

from autoplanner.utils.hash_utils import HashCounter
from autoplanner.__version__ import __version__


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
        :param node_hash: The node hashing function. Should take exactly 1 argument.
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

        def new_edge_hash(pair):
            h = edge_hash(pair)
            return '{}_{}_{}'.format(pair[0].field_type.parent_id, h, pair[1].field_type.parent_id)

        def new_node_hash(n):
            return '{}_{}'.format(node_hash(n), n.field_type.parent_id)

        self._edge_hash = new_edge_hash
        self._node_hash = node_hash
        self._edge_counter = HashCounter(func=self._edge_hash)
        self._node_counter = HashCounter(func=self._node_hash)
        self._weights = {}
        self.is_cached = False

    @property
    def plans(self):
        """Returns list of plans to compute (limited to depth)"""
        return list(self._plans)[-self.depth:]

    @plans.setter
    def plans(self, plans):
        """Sets the plans. """
        self._plans = plans
        self.is_cached = False

    @property
    def depth(self):
        return self._depth

    @depth.setter
    def depth(self, val):
        """Sets the depth"""
        self._depth = val
        self.is_cached = False

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

        self.cache_plans(plans)

        wires = self.collect_wires(plans)
        self._info("  {} wires loaded".format(len(wires)))

        operations = self.collect_operations(plans)
        self._info("  {} operations loaded".format(len(operations)))

        self.cache_wires(wires)
        edges = self.to_edges(wires, operations)
        self.update_tally(edges)

        self.plans += plans
        self.edges += edges
        self.save_weights(self.edges)

    def compute(self):
        """Compute the weights. If previously computed, this function will avoid re-caching plans and wires."""
        self._info("Computing weights for {} plans".format(len(self.plans)))
        if not self.is_cached:
            self.cache_plans(self.plans)
        else:
            self._info("   Plans already cached. Skipping...")
        wires = self.collect_wires(self.plans)
        self._info("  {} wires loaded".format(len(wires)))
        operations = self.collect_operations(self.plans)
        self._info("  {} operations loaded".format(len(operations)))
        if not self.is_cached:
            self.cache_wires(wires)
        else:
            self._info("   Wires already cached. Skipping...")
        self.is_cached = True
        self.edges = self.to_edges(wires, operations)
        self.update_tally(self.edges)
        self.save_weights(self.edges)

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
        self._info("   Caching {} wires...".format(len(wires)))
        self.browser.recursive_retrieve(wires[:], {
            "source": {
                "field_type": [],
                "operation": "operation_type",
                "allowable_field_type": {
                    "field_type"
                }
            },
            "destination": {
                "field_type": [],
                "operation": "operation_type",
                "allowable_field_type": {
                    "field_type"
                }
            }
        }, strict=False)

    def cache_plans(self, plans):
        self._info("   Caching plans...")
        self.browser.recursive_retrieve(plans, {
            "operations": {
                "field_values": {
                    "allowable_field_type": {
                        "field_type"
                    },
                    "field_type": [],
                    "wires_as_source": [],
                    "wires_as_dest": []
                }
            }
        }, strict=False)

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

        # due to DB inconsitencies, some wires and operations have not AFTS
        edges = [(n1, n2) for n1, n2 in edges if n1 is not None and n2 is not None]
        return edges

    def calculate_weights(self, edges):
        edge_counter = HashCounter(func=self._edge_hash)
        node_counter = HashCounter(func=self._node_hash)

        rows = []
        for n1, n2 in edges:
            if n1 and n2:
                rows.append({
                    "source": n1.id,
                    "destination": n2.id,
                    "count": edge_counter[(n1, n2)],
                    "total": node_counter[n1],
                })
        df = pd.DataFrame(rows)
        df.drop_duplicates(inplace=True)
        df['weight'] = df['count'] / df['total']
        df.sort_values(by=['weight'], inplace=True, ascending=True)
        return df

    def update_tally(self, edges):
        self._info("Hashing and counting edges...")
        for n1, n2 in tqdm(edges, desc="counting edges"):
            if n1 and n2:
                self._edge_counter[(n1, n2)] += 1
                self._node_counter[n2] += 1

    def save_weights(self, edges):
        for n1, n2 in edges:
            if n1 and n2:
                self._weights[self._edge_hash((n1, n2))] = self.cost(n1, n2)

    def cost(self, n1, n2):
        n = self._edge_counter[(n1, n2)] * 1.0
        t = self._node_counter[n1] * 1.0
        return self._cost_function(n, t)

    def default_cost_function(self, n, t):
        p = 10e-6
        if t > 0:
            p = n / t
        w = (1 - p) / (1 + p)
        return 10 / (1.000001 - w)

    def get_weight(self, n1, n2):
        if not self.is_cached:
            raise Exception("The tally and weights have not been computed")
        ehash = self._edge_hash((n1, n2))
        return self._weights.get(ehash, self.cost(n1, n2))

    def make_df(self):
        edges = self.edges
        counter = HashCounter(func=self._edge_hash)
        node_counter = HashCounter(func=self._node_hash)
        for n1, n2 in edges:
            counter[(n1, n2)] += 1
            node_counter[n1] += 1

        rows = []
        for n1, n2 in edges:
            if n1 and n2:
                rows.append({
                    "source": "{}_{}".format(n1.id, n1.field_type.operation_type.name),
                    "destination": "{}_{}".format(n2.id, n2.field_type.operation_type.name),
                    "count": counter[(n1, n2)],
                    "total": node_counter[n2],
                })
        df = pd.DataFrame(rows)
        df.drop_duplicates(inplace=True)
        df['probability'] = df['count'] / df['total']
        df.sort_values(by=['probability'], inplace=True, ascending=True)
        return df

    # def heatmap(self):
    #     df = self.make_df()
    #     sns.set()
    #     f, ax = plt.subplots(figsize=(12,9))
    #     sns.heatmap(df, annot=False, ax=ax, cmap="YlGnBu")


class BrowserGraph(object):
    """Graph class for representing Aquarium model-to-model relationships."""

    class DEFAULTS:

        MODEL_TYPE = "model"
        NODE_TYPE = "node"

    def __init__(self, browser):
        self.browser = browser
        self.graph = nx.DiGraph()
        self.model_hashes = {}
        self.prefix = ''
        self.suffix = ''

    def node_id(self, model):
        """
        Convert a pydent model into a unique graph id

        :param model: Trident model
        :type model: ModelBase
        :return: unique graph id for model
        :rtype: basestring
        """
        model_class = model.__class__.__name__
        model_hash = self.model_hashes.get(
            model_class,
            lambda model: "{cls}_{mid}".format(cls=model.__class__.__name__, mid=model.id)
        )
        return self.prefix + model_hash(model) + self.suffix

    def set_prefix(self, prefix):
        mapping = {n: prefix + n for n in self.nodes}
        self.graph = nx.relabel_nodes(self.graph, mapping)
        self.prefix = prefix

    def set_suffix(self, suffix):
        mapping = {n: n + suffix for n in self.nodes}
        self.graph = nx.relabel_nodes(self.graph, mapping)
        self.suffix = suffix

    # def add_special_node(self, node_id, node_class):
    #     self.graph.add_node(node_id, node_class=node_class, model_id=None)
    #
    # def get_special_node(self, node_id):
    #     return self.graph.node(node_id)

    def add_special_node(self, node_id, node_class):
        return self.graph.add_node(node_id, node_class=node_class, node_type=self.DEFAULTS.NODE_TYPE)

    def add_node(self, model, node_id=None):
        """
        Add a model node to the graph with optional node_id

        :param model: Trident model
        :type model: ModelBase
        :param node_id: optional node_id to use
        :type node_id: basestring
        :return: None
        :rtype: None
        """
        model_class = model.__class__.__name__
        if not issubclass(type(model), TridentBase):
            raise TypeError("Add node expects a Trident model, not a {}".format(type(model)))
        if node_id is None:
            node_id = self.node_id(model)
        return self.graph.add_node(node_id, node_class=model_class, model_id=model.id,
                                   node_type=self.DEFAULTS.MODEL_TYPE)

    def add_edge_from_models(self, m1, m2, edge_type=None, **kwargs):
        """
        Adds an edge from two models.

        :param m1:
        :type m1:
        :param m2:
        :type m2:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        self.add_edge(self.node_id(m1), self.node_id(m2), model1=m1, model2=m2, edge_type=edge_type, **kwargs)

    def add_edge(self, n1, n2, model1=None, model2=None, edge_type=None, **kwargs):
        """
        Adds edge between two nodes given the node ids. Raises error if node does not exist
        and models are not provided. If node_id does not exist and model is provided, a new
        node is added.

        :param n1: first node id
        :type n1: int
        :param n2: second node id
        :type n2: int
        :param model1: first model (optional)
        :type model1: ModelBase
        :param model2: second model (optional)
        :type model2: ModelBase
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        if n1 not in self.graph:
            if model1 is not None:
                self.add_node(model1, n1)
            else:
                raise Exception("Model1 must be provided")

        if n2 not in self.graph:
            if model1 is not None:
                self.add_node(model2, n2)
            else:
                raise Exception("Model2 must be provided")

        return self.graph.add_edge(n1, n2, edge_type=edge_type, **kwargs)

    def update_node(self, node_id, data):
        node = self.get_node(node_id)
        node.update(data)
        return node

    def predecessors(self, node_id):
        return self.graph.predecessors(node_id)

    def successors(self, node_id):
        return self.graph.successors(node_id)

    @classmethod
    def _convert_id(cls, n):
        if isinstance(n, int) or isinstance(n, str):
            return n
        elif issubclass(type(n), TridentBase):
            return cls.node_id(n)
        else:
            raise TypeError("Type '{}' {} not recognized as a node".format(type(n), n))

    def get_node(self, node_id):
        """
        Get a node from a node_id. If provided with Trident model, model is converted into
        a node_id.
        :param node_id:
        :type node_id:
        :return:
        :rtype:
        """

        node = self.graph.node[node_id]
        if 'model_id' in node and 'model' not in node:
            model = self.browser.find(node['model_id'], model_class=node['node_class'])
            node['model'] = model
        return node

    def get_model(self, node_id):
        node = self.get_node(node_id)
        return node.get('model', None)

    def get_edge(self, n1, n2):
        return self.graph.edges[n1, n2]

    def models(self, model_class=None):
        return list(self.iter_models(model_class))

    @property
    def nodes(self):
        return self.graph.nodes

    @property
    def edges(self):
        return self.graph.edges

    def iter_node_data(self, node_type=None, nbunch=None):
        if nbunch is None:
            nbunch = self
        for n in nbunch:
            node = self.get_node(n)
            if node_type is None or node['node_type'] == node_type:
                yield (n, node)

    def iter_edge_data(self, ebunch=None):
        if ebunch is None:
            ebunch = self
        for e in ebunch:
            yield (e, self.get_edge(*e))

    def iter_model_data(self, model_class=None, nbunch=None, **attrs):
        for n, ndata in self.iter_node_data(node_type=self.DEFAULTS.MODEL_TYPE, nbunch=nbunch):
            if model_class is None or ndata['node_class'] == model_class:
                model = ndata['model']
                passes = True
                for attr, val in attrs.items():
                    if not hasattr(model, attr) or getattr(model, attr) != val:
                        passes = False
                        break
                if passes:
                    yield (n, ndata)

    def iter_models(self, model_class=None, nbunch=None, **attrs):
        for n, ndata in self.iter_model_data(model_class, nbunch=nbunch, **attrs):
            yield ndata['model']

    @staticmethod
    def copy_graph(graph):
        graph_copy = nx.DiGraph()
        graph_copy.add_nodes_from(graph.nodes(data=True))
        graph_copy.add_edges_from(graph.edges(data=True))
        return graph_copy

    @classmethod
    def _array_to_identifiers(cls, nodes):
        formatted_nodes = []
        for n in nodes:
            if isinstance(n, int) or isinstance(n, str):
                formatted_nodes.append(n)
            elif issubclass(type(n), TridentBase):
                formatted_nodes.append(cls.node_id(n))
            else:
                raise TypeError("Type '{}' {} not recognized as a node".format(type(n), n))
        return formatted_nodes

    def subgraph(self, nodes):
        nodes = self._array_to_identifiers(nodes)

        graph_copy = nx.DiGraph()
        graph_copy.add_nodes_from((n, self.nodes[n]) for n in nodes)

        edges = []
        for n1, n2 in self.edges:
            if n1 in nodes and n2 in nodes:
                edges.append((n1, n2, self.edges[n1, n2]))

        graph_copy.add_edges_from(edges)

        browser_graph_copy = self.copy()
        browser_graph_copy.graph = graph_copy
        return browser_graph_copy

    def filter(self, key=None):
        if key is None:
            key = lambda x: True
        nodes = [n for n in self.nodes if key(n)]
        return self.subgraph(nodes)

    def remove(self, key=None):
        if key is None:
            key = lambda x: False
        nodes = [n for n in self.nodes if key(n)]
        return self.subgraph(set(self.graph.nodes).difference(set(nodes)))

    def only_models(self, model_class=None, **attrs):
        return self.subgraph([n for n, _ in self.iter_model_data(model_class=model_class, **attrs)])

    def filter_out_models(self, model_class=None, key=None):
        node_set = set(self.nodes)
        if key:
            for_removal = set()
            for n, ndata in self.iter_model_data(model_class=model_class):
                if key(ndata['model']):
                    for_removal.add(n)
        return self.subgraph(node_set.difference(for_removal))

    def cache_models(self):
        models = {}
        for n in self:
            ndata = self.graph.node[n]
            if 'model_id' in ndata:
                models.setdefault(ndata['node_class'], []).append(ndata['model_id'])

        models_by_id = {}
        for model_class, model_ids in models.items():
            found_models = self.browser.find(model_ids, model_class=model_class)
            models_by_id.update({m.id: m for m in found_models})
        for n in self:
            ndata = self.graph.node[n]
            if 'model_id' in ndata:
                ndata['model'] = models_by_id[ndata['model_id']]

    def copy(self):
        return self.__copy__()

    def roots(self):
        roots = []
        for n in self.graph:
            if not len(list(self.graph.predecessors(n))):
                roots.append(n)
        return roots

    def leaves(self):
        leaves = []
        for n in self.graph:
            if not len(list(self.graph.successors(n))):
                leaves.append(n)
        return leaves

    def __len__(self):
        return len(self.graph)

    def __iter__(self):
        return self.graph.__iter__()

    def __copy__(self):
        copied = self.__class__(self.browser)
        copied.graph = self.copy_graph(self.graph)
        copied.model_hashes = self.model_hashes
        return copied


class AutoPlannerModel(Loggable):
    """
    Builds a model from historical plan data.
    """

    def __init__(self, session, depth=100, plans=None):
        self.browser = Browser(session)
        self.weight_container = EdgeWeightContainer(self.browser, self.hash_afts, self.external_aft_hash,
                                                    depth=depth, plans=plans)
        self._template_graph = None
        self.model_filters = []
        self._version = __version__
        self.init_logger("AutoPlanner@{url}".format(url=session.url))

    @property
    def version(self):
        return self._version

    def set_plans(self, plans):
        self.weight_container.plans = plans
        self._template_graph = None

    def set_depth(self, depth):
        self.weight_container.depth = depth
        self._template_graph = None

    def set_verbose(self, verbose):
        if self.weight_container:
            self.weight_container.set_verbose(verbose)
        super().set_verbose(verbose)

    @staticmethod
    def external_aft_hash(aft):
        """A has function representing two 'external' :class:`pydent.models.AllowableFieldType`
        models (i.e. a wire)"""
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
        """A has function representing two 'internal' :class:`pydent.models.AllowableFieldType`
        models (i.e. an operation)"""

        return "{operation_type}".format(
            operation_type=aft.field_type.parent_id,
            routing=aft.field_type.routing,
            sample_type=aft.sample_type_id
        )

    @classmethod
    def hash_afts(cls, pair):
        """Make a unique hash for a :class:`pydent.models.AllowableFieldType` pair"""
        source_hash = cls.external_aft_hash(pair[0])
        dest_hash = cls.external_aft_hash(pair[1])
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

    @classmethod
    def match_internal_afts(cls, input_afts, output_afts):
        internal_groups = {}
        for aft in output_afts:
            internal_groups.setdefault(cls.internal_aft_hash(aft), []).append(aft)

        edges = []

        for iaft in input_afts:
            hsh = cls.internal_aft_hash(iaft)
            internals = internal_groups.get(hsh, [])
            for aft in internals:
                edges.append((iaft, aft))
        return edges

    @classmethod
    def match_external_afts(cls, input_afts, output_afts):
        external_groups = {}
        for aft in input_afts:
            external_groups.setdefault(cls.external_aft_hash(aft), []).append(aft)

        edges = []
        for oaft in output_afts:
            hsh = cls.external_aft_hash(oaft)
            externals = external_groups.get(hsh, [])
            for aft in externals:
                edges.append((oaft, aft))
        return edges

    @classmethod
    def match_afts(cls, input_afts, output_afts):
        return cls.match_internal_afts(input_afts, output_afts) + \
               cls.match_external_afts(input_afts, output_afts)

    def get_aft_pairs(self):
        """
        Construct edges from all deployed allowable_field_types

        :return: list of tuples representing connections between AllowableFieldType
        :rtype: list
        """
        input_afts, output_afts = self.cache_afts()
        return self.match_afts(input_afts, output_afts)

    @property
    def template_graph(self):
        if self._template_graph is None:
            self.construct_template_graph()
        graph = self._template_graph
        for model_class, filter_func in self.model_filters:
            graph = graph.filter_out_models(model_class=model_class, key=filter_func)
        return graph

    def add_model_filter(self, model_class, func):
        self.model_filters.append((model_class, func))

    def reset_model_filters(self):
        self.model_filter = []

    def add_weighted_edges(self, graph, weight_container):
        for aft1, aft2 in self.get_aft_pairs():

            edge_type = None
            roles = [aft.field_type.role for aft in [aft1, aft2]]
            if roles == ('input', 'output'):
                edge_type = 'internal'
            elif roles == ('output', 'input'):
                edge_type = 'external'

            graph.add_edge_from_models(
                aft1,
                aft2,
                edge_type=edge_type,
                weight=weight_container.get_weight(aft1, aft2)
            )

    def build(self):
        return self.construct_template_graph()

    def construct_template_graph(self):
        """
        Construct a graph of all possible Operation connections.
        """
        # computer weights
        self.weight_container.compute()

        self._info("Building Graph:")
        G = BrowserGraph(self.browser)
        # G.model_hashes = {
        #     "AllowableFieldType": lambda m: "{}_{}_{}_{}".format(m.__class__.__name__, m.object_type_id, m.sample_type_id, m.field_type_id)
        # }
        self.add_weighted_edges(G, self.weight_container)
        self._info("  {} edges".format(len(list(G.edges()))))
        self._info("  {} nodes".format(len(G)))

        self._template_graph = G
        return G

    def collect_afts(self, graph):
        """
        Collect :class:`pydent.models.AllowableFieldType` models from graph

        :param graph: a browser graph
        :type graph: BrowserGraph
        :return: list of tuples of input vs output allowable field types in the graph
        :rtype: list
        """
        afts = graph.models("AllowableFieldType")

        input_afts = [aft for aft in afts if aft.field_type.role == 'input']
        output_afts = [aft for aft in afts if aft.field_type.role == 'output']
        return input_afts, output_afts

    def print_path(self, path, graph):
        ots = []
        for n, ndata in graph.iter_model_data("AllowableFieldType", nbunch=path):
            aft = ndata['model']
            ot = self.browser.find(aft.field_type.parent_id, 'OperationType')
            ots.append("{ot} in '{category}'".format(category=ot.category, ot=ot.name))

        edge_weights = [graph.get_edge(x, y)['weight'] for x, y in zip(path[:-1], path[1:])]
        print("PATH: {}".format(path))
        print('WEIGHTS: {}'.format(edge_weights))
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
        for n, ndata in graph.iter_model_data("AllowableFieldType",
                                              object_type_id=obj1.id,
                                              sample_type_id=obj1.sample_type_id):
            graph.add_edge("START", n, weight=0)
        for n, ndata in graph.iter_model_data("AllowableFieldType",
                                              object_type_id=obj2.id,
                                              sample_type_id=obj2.sample_type_id):
            graph.add_edge(n, "END", weight=0)

        # find and sort shortest paths
        shortest_paths = []
        for n1 in graph.graph.successors("START"):
            n2 = "END"
            try:
                path = nx.dijkstra_path(graph.graph, n1, n2, weight='weight')
                path_length = nx.dijkstra_path_length(graph.graph, n1, n2, weight='weight')
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
            dill.dump({
                'browser': self.browser,
                'template_graph': self.template_graph,
                'version': self._version
            }, f)
        statinfo = stat(path)
        self._info("{} bytes written to '{}'".format(statinfo.st_size, path))

    def save(self, path):
        return self.dump(path)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = dill.load(f)
            browser = data['browser']
            ap = cls(browser.session)

            statinfo = stat(path)
            ap._info("{} bytes loaded from '{}' to new AutoPlanner (id={})".format(
                statinfo.st_size, path, id(ap)))

            ap.browser = browser
            ap._template_graph = data['template_graph']
            ap._version = data['version']
            return ap
