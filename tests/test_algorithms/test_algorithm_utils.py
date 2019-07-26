import networkx as nx
from terrarium.algorithms.utils import (
    input_groups,
    iter_root_items,
    iter_end_nodes,
    iter_root_operations,
)
from terrarium.utils.graph_utils import top_paths
import pytest
from terrarium.graphs import OperationGraph
from terrarium.builders import OperationGraphBuilder
from terrarium.utils.test_utils import timeit


class TestGraphUtils(object):
    def test_input_groups(self, graph):
        group_dict, groups = input_groups(graph)
        for k, v in group_dict.items():
            assert k in groups[v]

        for group in groups:
            for n in group:
                assert n in group_dict

    def test_top_paths(self):
        G = nx.path_graph(5, create_using=nx.DiGraph)
        assert top_paths(G, [0, 4], None)
        assert top_paths(G, [0, 1, 4], None)
        with pytest.raises(nx.exception.NetworkXNoPath):
            assert not top_paths(G, [1, 0, 4], None)
            assert not top_paths(G, [1, 5], None)

    def test_extract_root_operations(self, graph):
        nodes = list(iter_root_operations(graph))

    def test_extract_items(self, graph: OperationGraph, builder: OperationGraphBuilder):
        with timeit("Assigning inventory"):
            builder.assign_inventory(graph)
        nodes = list(iter_root_items(graph))
        print("Found {} items".format(len(nodes)))
        assert nodes

    def test_extract_end_nodes(self, session, graph):
        nodes = list(iter_end_nodes(graph, 1, 1))
