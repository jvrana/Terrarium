from terrarium.serializer import Serializer
from terrarium.utils import group_by
from terrarium.graphs import SampleGraph, AFTGraph, ModelGraph
from typing import Sequence
from .utils import match_afts
from .hashes import internal_aft_hash
import networkx as nx
from .requester import DataRequester
from .builder_abc import BuilderABC


class ProtocolGraphBuilder(BuilderABC):
    def __init__(
        self,
        requester: DataRequester,
        blueprint_graph: AFTGraph,
        sample_graph: SampleGraph,
    ):
        super().__init__(requester)
        self.blueprint_graph = blueprint_graph
        self.sample_graph = sample_graph

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
    def collect_sample_ids(cls, sample_graph: SampleGraph) -> Sequence[int]:
        sample_ids = [
            ndata["primary_key"] for _, ndata in sample_graph.model_data("Sample")
        ]
        return sorted(list(set(sample_ids)))

    def sample_subgraph(self, sample: dict) -> AFTGraph:
        nbunch = []
        for n, ndata in self.blueprint_graph.node_data():
            if ndata["sample_type_id"] == sample["sample_type_id"]:
                nbunch.append(n)
        subgraph = self.blueprint_graph.subgraph(nbunch)
        subgraph.add_prefix("Sample{}_".format(sample["primary_key"]))
        for _, ndata in subgraph.node_data():
            ndata["sample"] = sample
        return subgraph

    def build(self):
        return self.build_basic_graph()

    def build_basic_graph(self) -> ModelGraph:
        sample_graphs = {}
        for nid, sample_data in self.sample_graph.model_data("Sample"):
            sample_graphs[sample_data["primary_key"]] = self.sample_subgraph(
                sample_data
            )

        graph = ModelGraph()
        graph.schemas[0].update({"sample": dict})
        graph.graph = nx.compose_all([sg.graph for sg in sample_graphs.values()])

        for x in self.sample_graph.edges():
            s1 = self.sample_graph.get_data(x[0])
            s2 = self.sample_graph.get_data(x[1])

            g1 = sample_graphs[s1["primary_key"]]
            g2 = sample_graphs[s2["primary_key"]]
            edges = self.connect_sample_graphs(g1, g2)
            for e in edges:
                n1 = g1.node_id(e[0])
                n2 = g2.node_id(e[1])
                edge = self.blueprint_graph.get_edge(
                    self.blueprint_graph.node_id(e[0]),
                    self.blueprint_graph.node_id(e[1]),
                )
                graph.add_edge(
                    n1, n2, weight=edge["weight"], edge_type="sample_to_sample"
                )
        return graph

    def assign_items(self, graph, part_limit=50):
        afts = [ndata for _, ndata in graph.model_data("AllowableFieldType")]
        sample_ids = self.collect_sample_ids(self.sample_graph)
        item_data = self.requester.collect_items(afts, sample_ids)
        part_data = self.requester.collect_parts(sample_ids, lim=part_limit)

        items_by_object_type_id = group_by(item_data, key=lambda i: i["object_type_id"])

        def get_collection_type_id(part):
            return part.get("collections", [{}])[0].get("object_type_id", None)

        parts_by_collection = group_by(part_data, key=get_collection_type_id)
        parts_by_collection_by_sample = {
            k: group_by(v, key=lambda x: x["sample_id"])
            for k, v in parts_by_collection.items()
        }

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
