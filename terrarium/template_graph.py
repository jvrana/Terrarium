from itertools import chain, product
from more_itertools import flatten

from .utils import GroupCounter, group_by, dict_intersection
from .hashes import external_aft_hash, internal_aft_hash, edge_hash
from .model_graph import ModelGraph


class TemplateGraphBuilder(object):

    def __init__(self):
        self.edge_counter = GroupCounter()
        self.node_counter = GroupCounter()
        self.edge_counter.group(0, edge_hash)
        self.node_counter.group(0, external_aft_hash)
        self._template_graph = None

    @property
    def template_graph(self):
        return self._template_graph

    def build(self, all_nodes, nodes, edges):
        self.update_counters(nodes, edges)
        self._template_graph = self.build_template_graph(all_nodes)
        return self._template_graph

    def update_counters(self, node_data, edge_data):
        self.node_counter.update(node_data)
        self.edge_counter.update(edge_data)

    @staticmethod
    def match_afts(afts1, afts2, hash_function):
        group1 = group_by(afts1, hash_function)
        group2 = group_by(afts2, hash_function)

        d = dict_intersection(group1, group2, lambda a, b: product(a, b))
        edges = chain(*flatten((d.values())))
        return edges

    @staticmethod
    def cost_function(source_counts, edge_counts):
        s = source_counts
        e = edge_counts
        s = max(s, 0)
        e = max(e, 0)
        p = 10e-6
        if s > 0:
            p = e / s
        w = (1 - p) / (1 + p)
        return 10 / 1 - w

    def edge_cost(self, src, dest):
        e = self.edge_counter.get(0, (src, dest), default=0)
        n = self.node_counter.get(0, src, default=0)
        return self.cost_function(n, e)

    def build_template_graph(self, all_afts):
        input_afts = [aft for aft in all_afts if aft["field_type"]["role"] == "input"]
        output_afts = [aft for aft in all_afts if aft["field_type"]["role"] == "output"]

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
