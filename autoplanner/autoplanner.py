from copy import deepcopy
from functools import reduce
from uuid import uuid4

import networkx as nx
from pydent.browser import Browser
from pydent.utils import filter_list
from pydent.utils.logger import Loggable
from tqdm import tqdm
import dill

class EdgeCalculator(Loggable):
    DEFAULT_DEPTH = 100

    def __init__(self, browser, edge_hash, node_hash, depth):
        self.init_logger("EdgeCalculator@{}".format(browser.session.url))
        self.edge_hash = edge_hash
        self.node_hash = node_hash
        self.browser = browser
        self.edge_count = {}
        self.node_count = {}
        self.determine_historical_edge_weights(depth)

    def cache_wires_and_operations(self, depth):
        self._info("Caching wires and operations for {} Plans".format(depth))

        plans = self.browser.last(depth, "Plan")
        self.browser.recursive_retrieve(plans, {
            "operations": {"field_values": ["allowable_field_type", "wires_as_source", "wires_as_dest"]}}, strict=False)
        for p in plans:
            p.wires

        all_wires = reduce((lambda x, y: x + y), [p.wires for p in plans])
        all_operations = reduce((lambda x, y: x + y), [p.operations for p in plans])
        self.browser.recursive_retrieve(all_wires[:], {
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
        return {
            "wires": all_wires,
            "operations": all_operations
        }

    def determine_historical_edge_weights(self, depth):
        self._info("Determining edge weights")

        results = self.cache_wires_and_operations(depth)

        edge_count = {}
        node_count = {}

        edges = []
        for wire in results['wires']:
            if wire.source and wire.destination:
                edges.append((wire.source.allowable_field_type, wire.destination.allowable_field_type))
        for op in results['operations']:
            for i in op.inputs:
                for o in op.outputs:
                    edges.append((i.allowable_field_type, o.allowable_field_type))

        for n1, n2 in tqdm(edges, desc="computing edge weights"):
            if n1 and n2:
                ehash = self.edge_hash(n1, n2)
                edge_count.setdefault(ehash, 0)
                edge_count[ehash] += 1

                nhash = self.node_hash(n1)
                node_count.setdefault(nhash, 0)
                node_count[nhash] += 1

        self.edge_count = edge_count
        self.node_count = node_count

        self._info("  {} edges".format(len(self.edge_count)))
        self._info("  {} nodes".format(len(self.node_count)))

    def count_edges(self, n1, n2):
        return self.edge_count.get(self.edge_hash(n1, n2), 0)

    def count_nodes(self, n):
        return self.node_count.get(self.node_hash(n), 0)

    def cost_function(self, n1, n2):
        p = 1e-4

        n = self.count_edges(n1, n2) * 1.0
        t = self.count_nodes(n1) * 1.0

        if t > 0:
            p = n / t
        w = (1 - p) / (1 + p)
        return 10 / (1.0001 - w)


class AutoPlanner(Loggable):

    def __init__(self, session):
        self.browser = Browser(session)
        self.weight_calculator = None
        self.template_graph = None
        self.init_logger("AutoPlanner@{url}".format(url=session.url))

    def set_verbose(self, verbose):
        if self.weight_calculator:
            self.weight_calculator.set_verbose(verbose)
        super().set_verbose(verbose)

    @staticmethod
    def external_aft_hash(aft):
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
        return "{operation_type}".format(
            operation_type=aft.field_type.parent_id,
            routing=aft.field_type.routing,
            sample_type=aft.sample_type_id
        )

    @classmethod
    def hash_afts(cls, aft1, aft2):
        source_hash = cls.external_aft_hash(aft1)
        dest_hash = cls.external_aft_hash(aft2)
        return "{}->{}".format(source_hash, dest_hash)

    def cache_afts(self):
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

    def collect_edges(self):

        input_afts, output_afts = self.cache_afts()

        external_groups = {}
        for aft in input_afts:
            external_groups.setdefault(self.external_aft_hash(aft), []).append(aft)

        internal_groups = {}
        for aft in output_afts:
            internal_groups.setdefault(self.internal_aft_hash(aft), []).append(aft)

        # from tqdm import tqdm
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

    def compute_weights(self, depth):
        self.weight_calculator = EdgeCalculator(self.browser, self.hash_afts, self.external_aft_hash, depth=depth)

    def construct_template_graph(self, depth=100):
        self.compute_weights(depth=depth)

        G = nx.DiGraph()
        edges = self.collect_edges()
        for n1, n2 in edges:
            G.add_node(n1.id, model_class=n1.__class__.__name__, model=n1)
            G.add_node(n2.id, model_class=n2.__class__.__name__, model=n2)
            if n1 and n2:
                G.add_edge(n1.id, n2.id, weight=self.weight_calculator.cost_function(n1, n2))

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
        input_afts = []
        output_afts = []

        for n in graph.nodes:
            node = graph.node[n]
            if node['model_class'] == "AllowableFieldType":
                if node['model'].field_type.role == "input":
                    input_afts.append(node['model'])
                elif node['model'].field_type.role == 'output':
                    output_afts.append(node['model'])
        return {
            "input": input_afts,
            "output": output_afts
        }

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
        print("OP TYPES:\n".format(ots))

    def search_graph(self, goal_sample, goal_object_type, start_object_type):
        graph = self.copy_graph()

        # filter afts
        afts = self.collect_afts(graph)
        input_afts = afts['input']
        output_afts = afts['output']
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

    def pickle_browser(self, browser):
        browser_copy = deepcopy(self.browser)
        for k, v in browser_copy.model_cache:
            for model in v:
                model.session = None
        browser_copy.session = None
        return browser_copy

    def dump(self, path):
        with open(path, 'wb') as f:
            dill.dump(self, f)

    def load(self, path):
        with open(path, 'rb') as f:
            dill.load(f)
