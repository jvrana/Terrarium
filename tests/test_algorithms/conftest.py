from os.path import dirname, join, abspath, isfile
import pytest
from terrarium.graphs import OperationGraph
from terrarium.builders import OperationBlueprintBuilder, OperationGraphBuilder
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

            adapter = AquariumAdapter(sess)
            blueprint = OperationBlueprintBuilder(AquariumAdapter(sess))
            blueprint.collect_deployed()
            plans = adapter.session.Plan.last(500)
            blueprint.collect(plans)
            blueprint_graph = blueprint.build()

            s = example_sample
            with base_session.with_cache(timeout=60) as sess:
                adapter = AquariumAdapter(sess)
                sample_graph = adapter.build_sample_graph([s])

            builder = OperationGraphBuilder(
                AquariumAdapter(sess), blueprint_graph, sample_graph
            )

            graph = builder.build_basic_graph()
        sample_graph.write_gexf(join(data_path, "sample_graph.gexf"))

        graph.write(filepath)
        graph.write_gexf(join(data_path, "operation_graph.gexf"))

        blueprint_graph.write(join(data_path, "blueprint.json"))
        blueprint_graph.write_gexf(join(data_path, "blueprint.gexf"))
    else:
        graph = OperationGraph.read(filepath)
    return graph
