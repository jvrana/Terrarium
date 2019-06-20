from terrarium.utils import multi_group_by_key, multi_group_by
from terrarium.graphs import SampleGraph, AFTGraph, OperationGraph
from typing import Sequence
from .utils import match_afts
from .hashes import internal_aft_hash
import networkx as nx
from terrarium.adapters import AdapterABC
from .builder_abc import BuilderABC
from terrarium import constants as C


class OperationGraphBuilder(BuilderABC):
    def __init__(
        self,
        requester: AdapterABC,
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
            ndata[C.PRIMARY_KEY] for _, ndata in sample_graph.model_data("Sample")
        ]
        return sorted(list(set(sample_ids)))

    def sample_subgraph(self, sample: dict) -> AFTGraph:
        nbunch = []
        for n, ndata in self.blueprint_graph.node_data():
            if ndata["sample_type_id"] == sample["sample_type_id"]:
                nbunch.append(n)
        subgraph = self.blueprint_graph.subgraph(nbunch)
        subgraph.add_prefix("Sample{}_".format(sample[C.PRIMARY_KEY]))
        for _, ndata in subgraph.node_data():
            ndata["sample"] = sample
        return subgraph

    def sample_subgraph_dict(self):
        sample_graphs = {}
        for nid, sample_data in self.sample_graph.model_data("Sample"):
            sample_graphs[sample_data[C.PRIMARY_KEY]] = self.sample_subgraph(
                sample_data
            )
        return sample_graphs

    @staticmethod
    def has_invalid_internal_routing(d1: dict, d2: dict) -> bool:
        """
        Identifies afts that have invalid internal routing, that is wires
        that have the same sample_ids but different routing, or have the same
        routing but different sample_ids.

        :param d1:
        :param d2:
        :return: bool
        """
        if d1["field_type"]["role"] == "input" and d2["field_type"]["role"] == "output":
            routing1 = d1["field_type"]["routing"]
            routing2 = d2["field_type"]["routing"]
            sid1 = d1["sample"]["id"]
            sid2 = d2["sample"]["id"]
            if (sid1 == sid2) ^ (routing1 == routing2):
                return True
        return False

    @classmethod
    def clean(cls, graph: OperationGraph) -> OperationGraph:
        """
        Remove invalid wires from graph.

        :param graph:
        :type graph:
        :return:
        :rtype:
        """
        removal = []
        aft_nodes = [n for n, _ in graph.model_data("AllowableFieldType")]
        for n1, n2 in graph.edges:
            if n1 in aft_nodes and n2 in aft_nodes:
                ndata1 = graph.get_data(n1)
                ndata2 = graph.get_data(n2)
                if cls.has_invalid_internal_routing(ndata1, ndata2):
                    removal.append((n1, n2))
        graph.graph.remove_edges_from(removal)
        return graph

    def build(self):
        graph = self.build_basic_graph()
        self.assign_inventory(graph)

    def build_basic_graph(self) -> OperationGraph:
        sample_graphs = self.sample_subgraph_dict()

        graph = OperationGraph()
        graph.schemas[0].update({"sample": dict})
        composed = nx.compose_all([sg.graph for sg in sample_graphs.values()])
        graph.set_graph(composed, validate=False)

        for x in self.sample_graph.edges():
            s1 = self.sample_graph.get_data(x[0])
            s2 = self.sample_graph.get_data(x[1])

            g1 = sample_graphs[s1[C.PRIMARY_KEY]]
            g2 = sample_graphs[s2[C.PRIMARY_KEY]]
            edges = self.connect_sample_graphs(g1, g2)
            for e in edges:
                n1 = g1.node_id(e[0])
                n2 = g2.node_id(e[1])
                edge = self.blueprint_graph.get_edge(
                    self.blueprint_graph.node_id(e[0]),
                    self.blueprint_graph.node_id(e[1]),
                )
                graph.add_edge(
                    n1,
                    n2,
                    **{C.WEIGHT: edge[C.WEIGHT], C.EDGE_TYPE: C.SAMPLE_TO_SAMPLE}
                )
        return graph

    def assign_inventory(self, graph: OperationGraph, part_limit=50):
        afts = [ndata for _, ndata in graph.model_data("AllowableFieldType")]
        sample_ids = self.collect_sample_ids(self.sample_graph)
        item_data = self.requester.collect_items(afts, sample_ids)
        part_data = self.requester.collect_parts(sample_ids, lim=part_limit)

        item_dict = multi_group_by_key(item_data, keys=["object_type_id", "sample_id"])
        part_dict = multi_group_by(
            part_data,
            keyfuncs=[
                lambda x: x["collections"][0].get("object_type_id", None),
                lambda x: x["sample_id"],
            ],
        )

        new_nodes = []
        new_edges = []

        for aft in afts:
            sample_id = aft["sample"]["id"]
            object_type_id = aft["object_type_id"]
            if aft["field_type"]["part"]:
                data = part_dict
            else:
                data = item_dict

            try:
                items = data[object_type_id][sample_id]
            except KeyError:
                items = []
            for item in items:
                new_nodes.append(item)
                new_edges.append((item, aft))

        for n in new_nodes:
            graph.add_data(n)

        for item, node in new_edges:
            graph.add_edge_from_models(item, node, **{C.WEIGHT: 0})
