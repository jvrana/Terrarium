from terrarium.utils import GroupCounter
from .hashes import edge_hash, internal_aft_hash, external_aft_hash
from typing import Sequence
from terrarium.graphs import AFTGraph
from terrarium.builders.utils import match_afts
from terrarium.adapters.aquarium.requester import DataRequester
from .builder_abc import BuilderABC
from abc import abstractmethod


class BlueprintBuilderABC(BuilderABC):
    """
    A blueprint builder that constructs a graph of all possible deployed operations
    """

    @abstractmethod
    def collect(self):
        raise NotImplementedError

    @abstractmethod
    def update(self, data):
        raise NotImplementedError

    @abstractmethod
    def edge_cost(self, src: dict, dest: dict) -> float:
        raise NotImplementedError

    # TODO: change edge_type to a constant
    # TODO: change aft to something else like iof for iofilter
    def build_template_graph(self, all_nodes: Sequence[dict]) -> AFTGraph:
        input_afts = [aft for aft in all_nodes if aft["field_type"]["role"] == "input"]
        output_afts = [
            aft for aft in all_nodes if aft["field_type"]["role"] == "output"
        ]

        external_edges = match_afts(output_afts, input_afts, external_aft_hash)
        internal_edges = match_afts(input_afts, output_afts, internal_aft_hash)

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

    def build(self, num_plans):
        self.update(self.collect(num_plans))
        all_nodes = self.requester.collect_deployed_afts()
        return self.build_template_graph(all_nodes)


class OperationBlueprintBuilder(BlueprintBuilderABC):
    def __init__(self, requester: DataRequester):
        super().__init__(requester)
        self.edge_counter = GroupCounter()
        self.node_counter = GroupCounter()
        self.edge_counter.group(0, edge_hash)
        self.node_counter.group(0, external_aft_hash)

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

    def collect(self, num_plans):
        nodes, edges = self.requester.collect_afts_from_plans(num_plans)
        return nodes, edges

    def update(self, data):
        node_data, edge_data = data
        self.node_counter.update(node_data)
        self.edge_counter.update(edge_data)
