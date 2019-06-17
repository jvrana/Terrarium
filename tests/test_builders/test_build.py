import pytest
from terrarium import (
    Serializer,
    SampleGraphBuilder,
    ProtocolBlueprintBuilder,
    ProtocolGraphBuilder,
)
from terrarium.builders import DataRequester
import os
import networkx as nx
from terrarium.graphs import ModelGraph

EXAMPLE_SAMPLE_ID = 27608


@pytest.fixture(scope="module")
def sample_graph(base_session):
    session = base_session.with_cache(timeout=60)
    session.set_verbose(True)
    s = session.Sample.find(EXAMPLE_SAMPLE_ID)

    builder = SampleGraphBuilder(DataRequester(session))

    sample_graph = builder.build([s])
    yield sample_graph


@pytest.fixture(scope="module")
def blueprint_graph(base_session, sample_graph):
    with base_session.with_cache(timeout=60) as sess:
        blueprint = ProtocolBlueprintBuilder(DataRequester(sess)).build(30)
    return blueprint


@pytest.fixture(scope="module")
def basic_graph(base_session, sample_graph, blueprint_graph):
    sess = base_session.with_cache(timeout=60)
    builder = ProtocolGraphBuilder(DataRequester(sess), blueprint_graph, sample_graph)
    graph = builder.build_basic_graph()
    return graph, builder


class TestBuilds(object):
    def test_sample_graph_build(self, sample_graph, session):
        assert sample_graph

    def test_blueprint_graph(self, blueprint_graph):
        assert blueprint_graph

    def test_build_basic_graph(self, basic_graph):
        assert basic_graph[0]

    def test_assign_items(self, session, basic_graph):
        graph, builder = basic_graph
        builder.assign_items(graph, part_limit=50)


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
