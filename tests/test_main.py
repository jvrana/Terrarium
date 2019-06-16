import pytest
from terrarium import (
    Serializer,
    SampleGraphBuilder,
    ProtocolBlueprintBuilder,
    ProtocolGraphBuilder,
)
import os
import networkx as nx
from terrarium.graph import ModelGraph

EXAMPLE_SAMPLE_ID = 27608


@pytest.fixture(scope="module")
def sample_graph(base_session):
    session = base_session.with_cache(timeout=60)
    session.set_verbose(True)
    s = session.Sample.find(EXAMPLE_SAMPLE_ID)
    sample_graph = SampleGraphBuilder.build([s])
    yield sample_graph


@pytest.fixture(scope="module")
def blueprint_graph(base_session, sample_graph):

    with base_session.with_cache(timeout=60) as sess:
        plans = sess.Plan.last(30)
        nodes, edges = Serializer.serialize_plans(plans)

    with base_session.with_cache(timeout=60) as sess:
        all_afts = Serializer.serialize_all_afts(sess)
    blueprint = ProtocolBlueprintBuilder().build(
        all_nodes=all_afts, nodes=nodes, edges=edges
    )
    return blueprint


class TestBuilds(object):
    def test_sample_graph_build(self, sample_graph, session):
        assert sample_graph

    def test_blueprint_graph(self, blueprint_graph):
        assert blueprint_graph

    def test_build(self, session, blueprint_graph, sample_graph):
        graph = ProtocolGraphBuilder.build_graph(blueprint_graph, sample_graph)
        import networkx as nx

        print(nx.info(graph.graph))


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
