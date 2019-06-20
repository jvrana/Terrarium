import os
import networkx as nx
from terrarium.graphs import ModelGraph
from terrarium import constants as C


class TestBuilds(object):
    def test_sample_graph_build(self, sample_graph, session):
        assert sample_graph
        assert len(list(sample_graph.edges))

    def test_blueprint_graph(self, blueprint_graph):
        assert blueprint_graph
        assert len(list(blueprint_graph.edges))

    def test_build_basic_graph(self, basic_graph):
        assert basic_graph
        assert len(list(basic_graph.edges))

    def test_assign_items(self, graph_with_assigned_inventory):
        graph = graph_with_assigned_inventory
        model_classes = [ndata[C.MODEL_CLASS] for n, ndata in graph.model_data()]
        assert "Item" in model_classes

    def test_clean(self, basic_graph, graph_builder):
        graph_builder.clean(basic_graph)

    def test_build(self, graph_builder):
        graph_builder.build()


class TestGraphValidity(object):
    def check_num_inputs_vs_num_predecessors(self, g):
        for n, ndata in g.model_data(
            "AllowableFieldType", lambda x: x["field_type"]["role"] == "output"
        ):
            fts = ndata["field_type"]["operation_type"]["field_types"]
            inputs = [
                ft for ft in fts if ft["role"] == "input" and ft["ftype"] == "sample"
            ]

            num_inputs = len(inputs)
            num_predecessors = len(list(g.graph.predecessors(n)))
            assert num_inputs <= num_predecessors

    def test_valid_blueprint(self, blueprint_graph):
        self.check_num_inputs_vs_num_predecessors(blueprint_graph)

    def test_valid_basic_graph(self, basic_graph):
        self.check_num_inputs_vs_num_predecessors(basic_graph)

    def test_new_edges(self, basic_graph, graph_builder):
        sample_graph_dict = graph_builder.sample_subgraph_dict()

        for n1, n2 in graph_builder.sample_graph.edges:
            ndata1 = graph_builder.sample_graph.get_data(n1)
            ndata2 = graph_builder.sample_graph.get_data(n2)

            sg1 = sample_graph_dict[ndata1["id"]]
            sg2 = sample_graph_dict[ndata2["id"]]

            assert graph_builder.connect_sample_graphs(sg1, sg2)

    def test_has_sample_to_sample_edges(self, basic_graph):
        s2s = []
        for e1, e2, edata in basic_graph.edges.data():
            if edata[C.EDGE_TYPE] == C.SAMPLE_TO_SAMPLE:
                s2s.append((e1, e2))
        assert s2s


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
