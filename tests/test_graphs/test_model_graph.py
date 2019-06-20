import os
import random

import networkx as nx
import pytest
from terrarium.exceptions import SchemaValidationError
from terrarium.graphs import ModelGraph
from uuid import uuid4
from copy import copy
from terrarium import constants as C


def test_validation_error():
    model_graph = ModelGraph()
    with pytest.raises(SchemaValidationError):
        model_graph.add_data({C.PRIMARY_KEY: "hello"})


def build_random_graph(num_nodes, num_edges):
    node_ids = list(range(num_nodes))
    x = lambda i: {C.MODEL_CLASS: "Sample", C.PRIMARY_KEY: i}

    edges = []
    for _ in range(num_edges):
        edges.append(random.sample(node_ids, k=2))

    g = ModelGraph()
    for n in node_ids:
        g.add_data(x(n))
    for n1, n2 in edges:
        g.add_edge_from_models(x(n1), x(n2))
    return g


class TestMethods(object):
    def test_edge_data(self):
        g = build_random_graph(5, 3)

        for x in g.edges.data():
            print(x)


class TestCopyMethodsModelGraphs(object):

    parametrize_copy_function = pytest.mark.parametrize(
        "copy_function",
        [lambda x: x.copy(), lambda x: copy(x), lambda x: x.subgraph(x.nodes)],
        ids=["graph.copy()", "copy(graph)", "graph.subgraph(graph.nodes)"],
    )

    @parametrize_copy_function
    def test_copy_meta(self, copy_function):
        g = ModelGraph()
        name = str(uuid4())
        prefix = str(uuid4())
        suffix = str(uuid4())

        g.name = name
        g.prefix = prefix
        g.suffix = suffix
        g.graph.graph["nested"] = {"level1": {"level2": [1, 2, 3]}}
        g2 = copy_function(g)

        assert g2.prefix == prefix
        assert g2.suffix == suffix
        assert g2.name == name

    @parametrize_copy_function
    def test_copy_meta_nested_data(self, copy_function):
        g = ModelGraph()
        g.graph.graph["nested"] = {"level1": {"level2": [1, 2, 3]}}
        g2 = copy_function(g)
        assert g.graph.graph["nested"] is not g2.graph.graph["nested"]
        assert (
            g.graph.graph["nested"]["level1"] is not g2.graph.graph["nested"]["level1"]
        )
        assert (
            g.graph.graph["nested"]["level1"]["level2"]
            is not g2.graph.graph["nested"]["level1"]["level2"]
        )

    @parametrize_copy_function
    def test_copy_node_data(self, copy_function):
        g = ModelGraph()
        d = {
            C.MODEL_CLASS: "Sample",
            C.PRIMARY_KEY: 4,
            "id": 5,
            "list": [1, 2, 3],
            "nested_list": [[1]],
        }
        g.add_data(d)
        g2 = copy_function(g)

        data1 = g.get_data(g.node_id(d))
        data2 = g2.get_data(g2.node_id(d))

        assert data1["id"] == data2["id"]
        assert data1 is not data2
        assert data1["list"] == data2["list"]
        assert not data1["list"] is data2["list"]

        assert not data1["nested_list"][0] is data2["nested_list"][0]
        assert not data1["nested_list"] is data2["nested_list"]


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

    @pytest.mark.parametrize("num_edges", [1, 10, 20])
    @pytest.mark.parametrize("num_nodes", [2, 10, 20])
    def test_write(self, tmp_path, num_nodes, num_edges):
        g = build_random_graph(num_nodes, num_edges)
        path = str(tmp_path) + "graph.json"
        self.write_tester(g, path)

    @pytest.mark.parametrize("num_edges", [1, 10, 20])
    @pytest.mark.parametrize("num_nodes", [2, 10, 20])
    def test_read(self, tmp_path, num_nodes, num_edges):
        g = build_random_graph(num_nodes, num_edges)
        path = str(tmp_path) + "graph.json"
        self.read_tester(g, path)
