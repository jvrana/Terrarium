from terrarium.serializer import Serializer
from terrarium.graphs import SampleGraph, AFTGraph, ModelGraph
from typing import Sequence
from .utils import match_afts
from .hashes import internal_aft_hash
import networkx as nx
from collections import defaultdict


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
        matching_afts = match_afts(in_afts, out_afts, internal_aft_hash)
        return matching_afts

    @classmethod
    def sample_subgraph(cls, template_graph: AFTGraph, sample: dict) -> AFTGraph:
        nbunch = []
        for n, ndata in template_graph.node_data():
            if ndata["sample_type_id"] == sample["sample_type_id"]:
                nbunch.append(n)
        subgraph = template_graph.subgraph(nbunch)
        subgraph.add_prefix("Sample{}_".format(sample["primary_key"]))
        for _, ndata in subgraph.node_data():
            ndata["sample"] = sample
        return subgraph

    @classmethod
    def build_graph(
        cls, blueprint_graph: AFTGraph, sample_graph: SampleGraph
    ) -> ModelGraph:
        sample_graphs = {}
        for nid, sample_data in sample_graph.model_data("Sample"):
            sample_graphs[sample_data["primary_key"]] = cls.sample_subgraph(
                blueprint_graph, sample_data
            )

        graph = ModelGraph()
        graph.schemas[0].update({"sample": dict})
        graph.graph = nx.compose_all([sg.graph for sg in sample_graphs.values()])

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
                graph.add_edge(
                    n1, n2, weight=edge["weight"], edge_type="sample_to_sample"
                )
        return graph

    @classmethod
    def assign_items(cls, graph, browser, sample_ids):
        afts = [ndata for _, ndata in graph.model_data("AllowableFieldType")]
        non_part_afts = [aft for aft in afts if not aft["field_type"]["part"]]
        object_type_ids = list(set([aft["object_type_id"] for aft in non_part_afts]))

        items = browser.where(
            model_class="Item",
            query={"sample_id": sample_ids, "object_type_id": object_type_ids},
        )

        items_by_object_type_id = defaultdict(list)
        for item in items:
            items_by_object_type_id[item.object_type_id].append(item)

        part_by_sample_by_type = cls._find_parts_for_samples(
            browser, sample_ids, lim=50
        )

        new_nodes = []

        for node, ndata in graph.model_data("AllowableFieldType"):
            sample_id = ndata["sample"]["id"]
            if ndata["field_type"]["part"]:
                parts = part_by_sample_by_type.get(ndata["object_type_id"], {}).get(
                    sample_id, []
                )
                for part in parts[-1:]:
                    new_nodes.append(part)

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
