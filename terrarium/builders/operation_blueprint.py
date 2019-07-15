from terrarium.utils import GroupCounter
from .hashes import edge_hash, internal_aft_hash, external_aft_hash
from terrarium.graphs import AFTGraph
from terrarium.builders.utils import match_afts
from .builder_abc import BuilderABC
from abc import abstractmethod
from terrarium import constants as C
import networkx as nx
from itertools import chain
from terrarium.adapters import AdapterABC


class BlueprintException(Exception):
    pass


class BlueprintBuilderABC(BuilderABC):
    """
    A blueprint builder that constructs a graph of all possible deployed operations
    """

    def __init__(self, adapter: AdapterABC):
        assert hasattr(adapter, "collect_deployed_afts")
        assert hasattr(adapter, "collect_data_from_plans")
        super().__init__(adapter)
        self.collected_data = None
        self.deployed_nodes = None
        self.graph = None

    def collect(self, *args, **kwargs):
        self.collected_data = self.adapter.collect_io_values_from_plans(*args, **kwargs)

    def collect_deployed(self, *args, **kwargs):
        """
        Collect deployed allowable field types.

        :param args:
        :param kwargs:
        :return:
        """
        self.deployed_nodes = self.adapter.collect_deployed_afts(*args, **kwargs)

    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def edge_cost(self, src: dict, dest: dict) -> float:
        raise NotImplementedError

    def init_graph(self):
        all_nodes = self.deployed_nodes
        input_afts = [aft for aft in all_nodes if aft["field_type"]["role"] == "input"]
        output_afts = [
            aft for aft in all_nodes if aft["field_type"]["role"] == "output"
        ]

        external_edges = match_afts(output_afts, input_afts, external_aft_hash)
        internal_edges = match_afts(input_afts, output_afts, internal_aft_hash)

        graph = AFTGraph()
        for aft1, aft2 in chain(internal_edges, external_edges):
            graph.add_data(aft1)
            graph.add_data(aft2)
            graph.add_edge_from_models(
                aft1, aft2, **{C.WEIGHT: None, C.EDGE_TYPE: C.EXTERNAL_EDGE}
            )
        graph.to(nx.MultiDiGraph)
        return graph

    def populate_graph(self, graph, nodes, edges):
        pass

    def build_template_graph(self) -> AFTGraph:
        all_nodes = self.deployed_nodes
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
            graph.add_edge_from_models(
                aft1, aft2, **{C.WEIGHT: cost, C.EDGE_TYPE: C.EXTERNAL_EDGE}
            )

        for aft1, aft2 in internal_edges:
            cost = self.edge_cost(aft1, aft2)
            graph.add_data(aft1)
            graph.add_data(aft2)
            graph.add_edge_from_models(
                aft1, aft2, **{C.WEIGHT: cost, C.EDGE_TYPE: C.INTERNAL_EDGE}
            )
        self.graph = graph
        return graph

    def build(self):
        if not self.collected_data:
            raise BlueprintException(
                "Please run {} to build.".format(self.collect.__name__)
            )
        if not self.deployed_nodes:
            raise BlueprintException(
                "Please run {} to build.".format(self.collect_deployed.__name__)
            )
        self.update()
        return self.build_template_graph()


class OperationBlueprintBuilder(BlueprintBuilderABC):
    def __init__(self, adapter: AdapterABC):
        super().__init__(adapter)
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

    def update(self):
        node_data, edge_data = self.collected_data
        aft_nodes = [n["allowable_field_type"] for n in node_data]
        aft_edges = [
            (e1["allowable_field_type"], e2["allowable_field_type"])
            for e1, e2 in edge_data
        ]
        self.node_counter.update(aft_nodes)
        self.edge_counter.update(aft_edges)
