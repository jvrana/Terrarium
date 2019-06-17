from itertools import chain, product
from more_itertools import flatten
import networkx as nx

from typing import Sequence

from .utils import GroupCounter, group_by, dict_intersection, validate
from .hashes import external_aft_hash, internal_aft_hash, edge_hash
from .serializer import Serializer
from .graph import ModelGraph
from collections import defaultdict


class BuilderGraphBase(ModelGraph):

    SCHEMA = {}

    def __init__(self, name=None):
        super().__init__(name=name)
        self.schemas[0].update(self.SCHEMA)


class SampleGraph(BuilderGraphBase):

    SCHEMA = {"__class__": "Sample"}


class AFTGraph(BuilderGraphBase):

    SCHEMA = {
        "__class__": "AllowableFieldType",
        "object_type_id": validate.is_any_type_of(int, None),
        "sample_type_id": validate.is_any_type_of(int, None),
        "field_type": {"role": str, "part": bool, "ftype": str, "parent_id": int},
    }


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
            g = SampleGraph()

        model_serializer = Serializer.serialize

        sample_data_array = [model_serializer(s) for s in samples]
        sample_data_array = [
            d for d in sample_data_array if g.node_id(d) not in visited
        ]

        if sample_data_array:
            for sample_data in sample_data_array:
                node_id = g.node_id(sample_data)
                visited.add(node_id)
                g.add_data(sample_data)
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
        browser.get(parent_samples, {"field_values": "sample"})

        return cls.build(parent_samples, g=g, visited=visited)


class Utils(object):
    @staticmethod
    def match_afts(
        afts1: Sequence[dict], afts2: Sequence[dict], hash_function: callable
    ):
        group1 = group_by(afts1, hash_function)
        group2 = group_by(afts2, hash_function)

        d = dict_intersection(group1, group2, lambda a, b: product(a, b))
        edges = chain(*flatten((d.values())))
        return edges


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

    def update_counters(
        self, node_data: Sequence[dict], edge_data: Sequence[dict]
    ) -> None:
        self.node_counter.update(node_data)
        self.edge_counter.update(edge_data)

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

    def build(
        self, all_nodes: Sequence[dict], nodes: Sequence[dict], edges: Sequence[tuple]
    ) -> AFTGraph:
        self.update_counters(nodes, edges)
        self._template_graph = self.build_template_graph(all_nodes)
        return self._template_graph

    def build_template_graph(self, all_nodes: Sequence[dict]) -> AFTGraph:
        input_afts = [aft for aft in all_nodes if aft["field_type"]["role"] == "input"]
        output_afts = [
            aft for aft in all_nodes if aft["field_type"]["role"] == "output"
        ]

        external_edges = Utils.match_afts(output_afts, input_afts, external_aft_hash)
        internal_edges = Utils.match_afts(input_afts, output_afts, internal_aft_hash)

        graph = AFTGraph()

        for aft1, aft2 in external_edges:
            cost = self.edge_cost(aft1, aft2)
            graph.add_data(aft1)
            graph.add_data(aft2)
            graph.add_edge_from_models(aft1, aft2, weight=cost, edge_type="external")

        for aft1, aft2 in internal_edges:
            cost = self.edge_cost(aft1, aft2)
            graph.add_data(aft1)
            graph.add_data(aft2)
            graph.add_edge_from_models(aft1, aft2, weight=cost, edge_type="internal")
        return graph


class ProtocolGraphBuilder(object):
    @classmethod
    def connect_sample_graphs(cls, g1: SampleGraph, g2: SampleGraph) -> Sequence[tuple]:
        def collect_role(graph, role):
            return [
                ndata
                for n, ndata in graph.node_data()
                if ndata["field_type"]["role"] == role
            ]

        in_afts = collect_role(g1, "input")
        out_afts = collect_role(g2, "output")
        matching_afts = Utils.match_afts(in_afts, out_afts, internal_aft_hash)
        return matching_afts

    @classmethod
    def sample_type_subgraph(cls, template_graph: AFTGraph, stid: int) -> AFTGraph:
        nbunch = []
        for n, ndata in template_graph.node_data():
            if ndata["sample_type_id"] == stid:
                nbunch.append(n)
        return template_graph.subgraph(nbunch)

    @classmethod
    def build_graph(
        cls, blueprint_graph: AFTGraph, sample_graph: SampleGraph
    ) -> ModelGraph:
        sample_graphs = {}
        for nid, ndata in sample_graph.node_data():
            if ndata["__class__"] == "Sample":
                sample_id = ndata["primary_key"]
                stid = ndata["sample_type_id"]
                g = cls.sample_type_subgraph(blueprint_graph, stid)
                g.add_prefix("Sample{}_".format(sample_id))
                sample_graphs[sample_id] = g

        graph = ModelGraph()
        graph._graph = nx.compose_all([sg.graph for sg in sample_graphs.values()])

        for x in sample_graph.edges():
            s1 = sample_graph.get_data(x[0])
            s2 = sample_graph.get_data(x[1])

            g1 = sample_graphs[s1["primary_key"]]
            g2 = sample_graphs[s2["primary_key"]]
            edges = cls.connect_sample_graphs(g1, g2)
            for e in edges:
                n1 = g1.node_id(e[0])
                n2 = g2.node_id(e[1])
                edge = blueprint_graph.get_edge(
                    blueprint_graph.node_id(e[0]), blueprint_graph.node_id(e[1])
                )
                graph.nodes[n1]["sample"] = s1
                graph.nodes[n2]["sample"] = s2
                graph.add_edge(
                    n1, n2, weight=edge["weight"], edge_type="sample_to_sample"
                )
        return graph

    def assign_items(self, graph, browser, sample_ids, afts):
        non_part_afts = [aft for aft in afts if not aft.field_type.part]
        object_type_ids = list(set([aft.object_type_id for aft in non_part_afts]))

        items = browser.where(
            model_class="Item",
            query={"sample_id": sample_ids, "object_type_id": object_type_ids},
        )

        items_by_object_type_id = defaultdict(list)
        for item in items:
            items_by_object_type_id[item.object_type_id].append(item)

        for node, ndata in graph.model_data("AllowableFieldType"):
            sample = ndata["sample"]
            sample_type_id = ndata["sample_type_id"]

            # new_nodes = []
            # new_edges = []
            # for node, ndata in graph.iter_model_data(model_class="AllowableFieldType"):
            #     aft = ndata["model"]
            #     sample = ndata["sample"]
            #     if sample:
            #         sample_id = sample.id
            #         sample_type_id = sample.sample_type_id
            #     else:
            #         sample_id = None
            #         sample_type_id = None
            #     if aft.sample_type_id == sample_type_id:
            #         if aft.field_type.part:
            #             parts = part_by_sample_by_type.get(aft.object_type_id, {}).get(
            #                 sample_id, []
            #             )
            #             for part in parts[-1:]:
            #                 if part.sample_id == sample_id:
            #                     new_nodes.append(part)
            #                     new_edges.append((part, sample, node))
            #         else:
            #             items = items_by_object_type_id[aft.object_type_id]
            #             for item in items:
            #                 if item.sample_id == sample_id:
            #                     new_nodes.append(item)
            #                     new_edges.append((item, sample, node))
            #
            # for n in new_nodes:
            #     graph.add_node(n)
            #
            # for item, sample, node in new_edges:
            #     graph.add_edge(graph.node_id(item), node, weight=0)
            #
            # self._info(
            #     "{} items added to various allowable_field_types".format(len(new_edges))
            # )
            # return graph

    @staticmethod
    def _find_parts_for_samples(browser, sample_ids, lim=50):
        all_parts = []
        part_type = browser.find_by_name("__Part", model_class="ObjectType")
        for sample_id in sample_ids:
            sample_parts = browser.last(
                lim,
                model_class="Item",
                object_type_id=part_type.id,
                sample_id=sample_id,
            )
            all_parts += sample_parts
        browser.retrieve(all_parts, "collections")

        # filter out parts that do not exist
        all_parts = [
            part
            for part in all_parts
            if part.collections and part.collections[0].location != "deleted"
        ]

        # create a Part-by-Sample-by-ObjectType dictionary
        data = {}
        for part in all_parts:
            if part.collections:
                data.setdefault(part.collections[0].object_type_id, {}).setdefault(
                    part.sample_id, []
                ).append(part)
        return data

    # def assign_items(self):
    #
    #
    #     ##############################
    #     # Get items
    #     ##############################
    #
    #     # requires aft[field_type][part]
    #     # list of non-part items
    #     # items by object_type_id
    #     # part items from sample_ids
    #     # assign items to graph
    #
    #     non_part_afts = [aft for aft in afts if not aft.field_type.part]
    #     object_type_ids = list(set([aft.object_type_id for aft in non_part_afts]))
    #
    #     self._cinfo(
    #         "finding all relevant items for {} samples and {} object_types".format(
    #             len(sample_ids), len(object_type_ids)
    #         )
    #     )
    #     items = browser.where(
    #         model_class="Item",
    #         query={"sample_id":
    #
    #         , "object_type_id": object_type_ids},
    #     )
    #     items = [item for item in items if item.location != "deleted"]
    #     self._info("{} total items found".format(len(items)))
    #     items_by_object_type_id = defaultdict(list)
    #     for item in items:
    #         items_by_object_type_id[item.object_type_id].append(item)
    #
    #     ##############################
    #     # Get parts
    #     ##############################
    #
    #     self._cinfo("finding relevant parts/collections")
    #     part_by_sample_by_type = self._find_parts_for_samples(
    #         browser, sample_ids, lim=50
    #     )
    #     self._cinfo("found {} collection types".format(len(part_by_sample_by_type)))
    #
    #     ##############################
    #     # Assign Items/Parts/Collections
    #     ##############################
    #
    #     new_nodes = []
    #     new_edges = []
    #     for node, ndata in graph.iter_model_data(model_class="AllowableFieldType"):
    #         aft = ndata["model"]
    #         sample = ndata["sample"]
    #         if sample:
    #             sample_id = sample.id
    #             sample_type_id = sample.sample_type_id
    #         else:
    #             sample_id = None
    #             sample_type_id = None
    #         if aft.sample_type_id == sample_type_id:
    #             if aft.field_type.part:
    #                 parts = part_by_sample_by_type.get(aft.object_type_id, {}).get(
    #                     sample_id, []
    #                 )
    #                 for part in parts[-1:]:
    #                     if part.sample_id == sample_id:
    #                         new_nodes.append(part)
    #                         new_edges.append((part, sample, node))
    #             else:
    #                 items = items_by_object_type_id[aft.object_type_id]
    #                 for item in items:
    #                     if item.sample_id == sample_id:
    #                         new_nodes.append(item)
    #                         new_edges.append((item, sample, node))
    #
    #     for n in new_nodes:
    #         graph.add_node(n)
    #
    #     for item, sample, node in new_edges:
    #         graph.add_edge(graph.node_id(item), node, weight=0)
    #
    #     self._info(
    #         "{} items added to various allowable_field_types".format(len(new_edges))
    #     )
    #     return graph


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
