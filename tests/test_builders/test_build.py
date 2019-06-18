import os
import networkx as nx
from terrarium.graphs import ModelGraph


class TestBuilds(object):
    def test_sample_graph_build(self, sample_graph, session):
        assert sample_graph
        assert len(list(sample_graph.edges))

    def test_blueprint_graph(self, blueprint_graph):
        assert blueprint_graph
        assert len(list(blueprint_graph.edges))

    def test_build_basic_graph(self, basic_graph):
        assert basic_graph[0]
        assert len(list(basic_graph[0].edges))

    def test_assign_items(self, graph_with_assigned_inventory):
        graph, builder = graph_with_assigned_inventory
        model_classes = [ndata["__class__"] for n, ndata in graph.model_data()]
        assert "Item" in model_classes

    def test_build(self, basic_graph):
        graph, builder = basic_graph
        builder.build()


class TestReadWrite(object):
    def write_tester(self, model_graph, path):
        model_graph.write(str(path))
        assert os.path.isfile(path)
        return path

    def read_tester(self, model_graph, path):
        filepath = self.write_tester(model_graph, path)
        loaded_graph = ModelGraph.read(str(filepath))
        assert nx.info(model_graph.graph) == nx.info(loaded_graph.graph)

    def test_sample_graph_save(self, sample_graph, tmp_path):
        """Expect a file to be written to the temporary path"""
        filepath = tmp_path / "sample_graph.test.json"
        self.write_tester(sample_graph, filepath)
        self.write_tester(sample_graph, "sample_graph.json")

    def test_sample_graph_load(self, sample_graph, tmp_path):
        """We expect the loaded nx.DiGraph to be equivalent to the """
        filepath = tmp_path / "sample_graph.test.json"
        self.read_tester(sample_graph, filepath)

    def test_blueprint_save(self, blueprint_graph, tmp_path):
        filepath = tmp_path / "blueprint_graph.test.json"
        self.write_tester(blueprint_graph, filepath)

    def test_blueprint_read(self, blueprint_graph, tmp_path):
        filepath = tmp_path / "blueprint_graph.test.json"
        self.read_tester(blueprint_graph, filepath)
