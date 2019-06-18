import pytest
from terrarium.builders import (
    SampleGraphBuilder,
    OperationBlueprintBuilder,
    OperationGraphBuilder,
)
from terrarium.adapters import DataRequester


EXAMPLE_SAMPLE_ID = 27608


@pytest.fixture(scope="module")
def example_sample(base_session):
    return base_session.Sample.find(EXAMPLE_SAMPLE_ID)


@pytest.fixture(scope="module")
def sample_graph(base_session, example_sample):
    session = base_session.with_cache(timeout=60)
    session.set_verbose(True)
    s = example_sample

    builder = SampleGraphBuilder(DataRequester(session))

    sample_graph = builder.build([s])
    yield sample_graph


@pytest.fixture(scope="module")
def blueprint_graph(base_session, sample_graph):
    with base_session.with_cache(timeout=60) as sess:
        blueprint = OperationBlueprintBuilder(DataRequester(sess)).build(30)
    return blueprint


@pytest.fixture(scope="module")
def basic_graph(base_session, sample_graph, blueprint_graph):
    sess = base_session.with_cache(timeout=60)
    builder = OperationGraphBuilder(DataRequester(sess), blueprint_graph, sample_graph)
    graph = builder.build_basic_graph()
    return graph, builder


@pytest.fixture(scope="module")
def graph_with_assigned_inventory(basic_graph):
    graph, builder = basic_graph
    builder.assign_inventory(graph, part_limit=50)
    return graph, builder
