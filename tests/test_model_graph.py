import networkx as nx
from terrarium.graph import ModelGraph, SchemaValidationError
import os
import random
import pytest


def test_validation_error():

    model_graph = ModelGraph()
    with pytest.raises(SchemaValidationError):
        model_graph.add_data({"primary_key": "hello"})


class TestReadWriteModelGraphs(object):
    def write_tester(self, model_graph, path):
        model_graph.write(str(path))
        assert os.path.isfile(path)
        return path

    def read_tester(self, model_graph, path):
        filepath = self.write_tester(model_graph, path)
        loaded_graph = ModelGraph.read(str(filepath))
        assert nx.info(model_graph.graph) == nx.info(loaded_graph.graph)
        return loaded_graph

    def test_write_empty(self, tmp_path):
        g = ModelGraph()
        path = str(tmp_path) + "graph.json"
        self.write_tester(g, path)

    def test_read_empty(self, tmp_path):
        g = ModelGraph()
        path = str(tmp_path) + "graph.json"
        self.read_tester(g, path)

    def build_random_graph(self, num_nodes, num_edges):
        node_ids = list(range(num_nodes))
        x = lambda i: {"__class__": "Sample", "primary_key": i}

        edges = []
        for _ in range(num_edges):
            edges.append(random.sample(node_ids, k=2))

        g = ModelGraph()
        for n in node_ids:
            g.add_data(x(n))
        for n1, n2 in edges:
            g.add_edge_from_models(x(n1), x(n2))
        return g

    @pytest.mark.parametrize("num_edges", [1, 10, 20])
    @pytest.mark.parametrize("num_nodes", [2, 10, 20])
    def test_write(self, tmp_path, num_nodes, num_edges):
        g = self.build_random_graph(num_nodes, num_edges)
        path = str(tmp_path) + "graph.json"
        self.write_tester(g, path)

    @pytest.mark.parametrize("num_edges", [1, 10, 20])
    @pytest.mark.parametrize("num_nodes", [2, 10, 20])
    def test_read(self, tmp_path, num_nodes, num_edges):
        g = self.build_random_graph(num_nodes, num_edges)
        path = str(tmp_path) + "graph.json"
        self.read_tester(g, path)
