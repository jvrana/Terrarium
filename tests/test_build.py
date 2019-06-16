import pytest
from terrarium import (
    Serializer,
    SampleGraphBuilder,
    ProtocolBlueprintBuilder,
    ProtocolGraphBuilder,
)

EXAMPLE_SAMPLE_ID = 27608


@pytest.fixture(scope="module")
def sample_graph(base_session):
    session = base_session.with_cache(timeout=60)
    session.set_verbose(True)
    s = session.Sample.find(EXAMPLE_SAMPLE_ID)
    sample_graph = SampleGraphBuilder.build([s])
    yield sample_graph


def test_sample_graph_build(sample_graph, session):
    assert sample_graph


def test_sample_graph_save(sample_graph, tmp_path):
    sample_graph.write(str(tmp_path) + ".json")


def test_sample_graph_load(sample_graph, tmp_path):
    path = str(tmp_path) + '.json'
    sample_graph.write(path)
    sample_graph.read(path)


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


def test_blueprint_graph(blueprint_graph):
    assert blueprint_graph


def test_build(session, blueprint_graph, sample_graph):
    graph = ProtocolGraphBuilder.build_graph(blueprint_graph, sample_graph)
    import networkx as nx
    print(nx.info(graph.graph))