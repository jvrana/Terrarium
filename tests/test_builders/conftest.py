import pytest
from terrarium.builders import (
    SampleGraphBuilder,
    OperationBlueprintBuilder,
    OperationGraphBuilder,
)
from terrarium.adapters import AquariumAdapter
from copy import deepcopy

EXAMPLE_SAMPLE_ID = 27608


@pytest.fixture(scope="module")
def example_sample(base_session):
    return base_session.Sample.find(EXAMPLE_SAMPLE_ID)


@pytest.fixture(scope="module")
def sample_graph(base_session, example_sample):
    session = base_session.with_cache(timeout=60)
    session.set_verbose(True)
    s = example_sample

    builder = SampleGraphBuilder(AquariumAdapter(session))

    sample_graph = builder.build([s])
    yield sample_graph


@pytest.fixture(scope="module")
def blueprint_graph(base_session, sample_graph):
    with base_session.with_cache(timeout=60) as sess:
        blueprint = OperationBlueprintBuilder(AquariumAdapter(sess)).build(30)
    return blueprint


@pytest.fixture(scope="module")
def build_basic_graph(base_session, sample_graph, blueprint_graph):
    sess = base_session.with_cache(timeout=60)
    builder = OperationGraphBuilder(
        AquariumAdapter(sess), blueprint_graph, sample_graph
    )
    graph = builder.build_basic_graph()
    return graph, builder


@pytest.fixture(scope="module")
def basic_graph(build_basic_graph):
    return build_basic_graph[0]


@pytest.fixture(scope="module")
def graph_builder(build_basic_graph):
    return build_basic_graph[1]


@pytest.fixture(scope="module")
def graph_with_assigned_inventory(graph_builder, basic_graph):
    graph = deepcopy(basic_graph)
    graph_builder.assign_inventory(graph, part_limit=50)
    return graph
