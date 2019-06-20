from os.path import dirname, join, abspath, isfile
import pytest
from terrarium.graphs import OperationGraph
from terrarium.builders import (
    SampleGraphBuilder,
    OperationBlueprintBuilder,
    OperationGraphBuilder,
)
from terrarium.adapters import AquariumAdapter


EXAMPLE_SAMPLE_ID = 27608
TIMEOUT = 60
NUM_PLANS = 30
FILENAME = "operation_graph.json"
DIRNAME = "data"

data_path = join(dirname(abspath(__file__)), DIRNAME)
filepath = join(data_path, FILENAME)


EXAMPLE_SAMPLE_ID = 27608


@pytest.fixture
def example_sample(base_session):
    return base_session.Sample.find(EXAMPLE_SAMPLE_ID)


@pytest.fixture
def graph(base_session, example_sample):

    if not isfile(filepath):
        with base_session.with_cache(timeout=TIMEOUT) as sess:
            s = example_sample
            requester = AquariumAdapter(sess)

            sample_builder = SampleGraphBuilder(requester)
            blueprint_builder = OperationBlueprintBuilder(requester)

            sample_graph = sample_builder.build([s])
            blueprint = blueprint_builder.build(NUM_PLANS)

            builder = OperationGraphBuilder(
                AquariumAdapter(sess), blueprint, sample_graph
            )
            graph = builder.build_basic_graph()
            builder.assign_inventory(graph, part_limit=50)
        graph.write(filepath)
    else:
        graph = OperationGraph.read(filepath)
    return graph
