from itertools import chain, product
from more_itertools import flatten
import networkx as nx

from typing import Sequence

from .utils import GroupCounter, group_by, dict_intersection
from .hashes import external_aft_hash, internal_aft_hash, edge_hash
from .model_graph import ModelGraph
from .serializer import Serializer


class SampleGraphBuilder(object):

    @classmethod
    def build(cls, samples, g=None, visited=None):
        """
        Requires requests.

        :param samples:
        :type samples:
        :param g:
        :type g:
        :param visited:
        :type visited:
        :return:
        :rtype:
        """
        if visited is None:
            visited = set()
        if g is None:
            g = ModelGraph()

        model_serializer = Serializer.serialize

        sample_data_array = [model_serializer(s) for s in samples]
        sample_data_array = [d for d in sample_data_array if g.node_id(d) not in g]

        if sample_data_array:
            for sample_data in sample_data_array:
                node_id = g.node_id(sample_data)
                visited.add(node_id)
                g.add_model(sample_data)
        else:
            return g

        parent_samples = []
        for sample in samples:
            for fv in sample.field_values:
                if fv.sample:
                    parent_samples.append(fv.sample)
                    m1 = model_serializer(sample)
                    m2 = model_serializer(fv.sample)
                    g.add_edge_from_models(m1, m2)

        browser = sample.session.browser
        browser.get(parent_samples, {
            'field_values': 'sample'
        })

        return cls.build(parent_samples, g=g, visited=visited)


class ProtocolBlueprintBuilder(object):

    def __init__(self):
        self.edge_counter = GroupCounter()
        self.node_counter = GroupCounter()
        self.edge_counter.group(0, edge_hash)
        self.node_counter.group(0, external_aft_hash)
        self._template_graph = None

    @property
    def template_graph(self):
        return self._template_graph

    def update_counters(self, node_data: Sequence[dict], edge_data: Sequence[dict]) -> None:
        self.node_counter.update(node_data)
        self.edge_counter.update(edge_data)

    @staticmethod
    def match_afts(afts1: Sequence[dict], afts2: Sequence[dict], hash_function: callable):
        group1 = group_by(afts1, hash_function)
        group2 = group_by(afts2, hash_function)

        d = dict_intersection(group1, group2, lambda a, b: product(a, b))
        edges = chain(*flatten((d.values())))
        return edges

    @staticmethod
    def cost_function(source_counts: int, edge_counts: int) -> float:
        s = source_counts
        e = edge_counts
        s = max(s, 0)
        e = max(e, 0)
        p = 10e-6
        if s > 0:
            p = e / s
        w = (1 - p) / (1 + p)
        return 10 / 1 - w

    def edge_cost(self, src: dict, dest: dict) -> float:
        e = self.edge_counter.get(0, (src, dest), default=0)
        n = self.node_counter.get(0, src, default=0)
        return self.cost_function(n, e)

    def build(self, all_nodes: Sequence[dict], nodes: Sequence[dict], edges: Sequence[tuple]) -> ModelGraph:
        self.update_counters(nodes, edges)
        self._template_graph = self._build_template_graph(all_nodes)
        return self._template_graph

    def build_template_graph(self, all_nodes: Sequence[dict]) -> ModelGraph:
        input_afts = [aft for aft in all_nodes if aft["field_type"]["role"] == "input"]
        output_afts = [aft for aft in all_nodes if aft["field_type"]["role"] == "output"]

        external_edges = self.match_afts(output_afts, input_afts, external_aft_hash)
        internal_edges = self.match_afts(input_afts, output_afts, internal_aft_hash)

        graph = ModelGraph()

        for aft1, aft2 in external_edges:
            cost = self.edge_cost(aft1, aft2)
            graph.add_model(aft1)
            graph.add_model(aft2)
            graph.add_edge_from_models(aft1, aft2, weight=cost, edge_type="external")

        for aft1, aft2 in internal_edges:
            cost = self.edge_cost(aft1, aft2)
            graph.add_model(aft1)
            graph.add_model(aft2)
            graph.add_edge_from_models(aft1, aft2, weight=cost, edge_type="internal")
        return graph

    @classmethod
    def sample_type_subgraph(cls, template_graph: ModelGraph, stid: int) -> ModelGraph:
        graph = template_graph.graph
        nbunch = []
        for n in graph:
            ndata = graph[n]
            if ndata['sample_type_id'] == stid:
                nbunch.append(n)
        return template_graph.subgraph(nbunch)


class ProtocolGraphBuilder(object):

    @classmethod
    def connect_sample_graphs(cls, g1: ModelGraph, g2: ModelGraph) -> Sequence[tuple]:
        def collect_role(graph, role):
            return [ndata for n, ndata in graph.nodes(data=True) if ndata['field_type']['role'] == role]
        out_afts = collect_role(g1.graph, 'output')
        in_afts = collect_role(g2.graph, 'input')
        edges = cls.match_afts(in_afts, out_afts, internal_aft_hash)
        return edges

    @classmethod
    def build_graph(cls, template_graph: ModelGraph, sample_graph: ModelGraph) -> ModelGraph:
        sample_graphs = {}
        for sample in sample_graph.iter_models(model_class="Sample"):
            g = cls.sample_type_subgraph(template_graph, sample.sample_type_id)
            g.set_prefix("Sample{}_".format(sample.id))
            sample_graphs[sample.id] = g

        graph = ModelGraph(template_graph.browser)
        graph._graph = nx.compose_all([sg.graph for sg in sample_graphs.values()])

        for x in sample_graph.edges():
            s1 = sample_graph.get_model(x[0])
            s2 = sample_graph.get_model(x[1])

            g1 = sample_graphs[s1.id]
            g2 = sample_graphs[s2.id]
            edges = cls.connect_sample_graphs(g1, g2)
            for e in edges:
                n1 = template_graph.node_id(e[0])
                n2 = template_graph.node_id(e[1])
                edge = template_graph.get_edge(
                    template_graph.node_id(e[0]), template_graph.node_id(e[1])
                )
                graph.nodes[n1]["sample"] = s1
                graph.nodes[n2]["sample"] = s2
                graph.add_edge(
                    n1, n2, weight=edge["weight"], edge_type="sample_to_sample"
                )
        return graph

#
#
# def cost_function(source_counts, edge_counts):
#     s = source_counts
#     e = edge_counts
#     s = max(s, 0)
#     e = max(e, 0)
#     p = 10e-6
#     if s > 0:
#         p = e / s
#     w = (1 - p) / (1 + p)
#     return 10 / 1 - w
#
#
# def edge_cost(src, dest, node_counter, edge_counter):
#     e = edge_counter.get(0, (src, dest), default=0)
#     n = node_counter.get(0, src, default=0)
#     return cost_function(n, e)
#
#
# # only method that requires Trident
# def collect_aft_data(session):
#     session.OperationType.where({"deployed": True})
#
#     session.browser.get('OperationType', {
#         'field_types': {
#             'allowable_field_types': {
#                 'object_type',
#                 'sample_type',
#                 'field_type'
#             }
#         }
#     })
#     afts = [aft_data(x) for x in session.browser.get('AllowableFieldType')]
#     return afts
#
#
# def build_template_graph(afts, node_counter, edge_counter):
#     input_afts = [aft for aft in afts if aft['role'] == 'input']
#     output_afts = [aft for aft in afts if aft['role'] == 'output']
#
#     external_edges = match_afts(output_afts, input_afts, external_aft_hash)
#     internal_edges = match_afts(input_afts, output_afts, internal_aft_hash)
#
#     graph = nx.DiGraph()
#
#     for aft1, aft2 in external_edges:
#         cost = edge_cost(aft1, aft2, node_counter=node_counter, edge_counter=edge_counter)
#         graph.add_node(node_id(aft1), data=aft1)
#         graph.add_node(node_id(aft1), data=aft2)
#         graph.add_edge(node_id(aft1), node_id(aft2), weight=cost, edge_type='external')
#
#     for aft1, aft2 in internal_edges:
#         cost = edge_cost(aft1, aft2, node_counter=node_counter, edge_counter=edge_counter)
#         graph.add_node(node_id(aft1), data=aft1)
#         graph.add_node(node_id(aft1), data=aft2)
#         graph.add_edge(node_id(aft1), node_id(aft2), weight=cost, edge_type='internal')
#
#     return graph
#
#
# plans = session.Plan.last(30)
# node_counter, edge_counter = build_counters(plans)
# afts = collect_aft_data(session)
# template_graph = build_template_graph(afts,
#                                       node_counter=node_counter,
#                                       edge_counter=edge_counter)
