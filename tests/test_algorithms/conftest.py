"""
Fixtures for algorithm tests.

Produces a graphs in './data' as .gexf files if None exists. Otherwise loads the gexf files for other tests.
"""

from os.path import dirname, join, abspath, isfile, isdir
from os import makedirs
import pytest
from terrarium.graphs import OperationGraph, AFTGraph, SampleGraph
from terrarium.builders import OperationBlueprintBuilder, OperationGraphBuilder
from terrarium.adapters import AquariumAdapter
from terrarium.utils.test_utils import timeit


EXAMPLE_SAMPLE_ID = 27608
TIMEOUT = 60
NUM_PLANS = 30
FILENAME = "operation_graph.json"
DIRNAME = "data"

data_path = join(dirname(abspath(__file__)), DIRNAME)
if not isdir(data_path):
    makedirs(data_path)
filepath = join(data_path, FILENAME)
blueprintgraph_filepath = join(data_path, "blueprint.json")
samplegraph_filepath = join(data_path, "sample_graph.json")


EXAMPLE_SAMPLE_ID = 27608


@pytest.fixture
def example_sample(base_session):
    return base_session.Sample.find(EXAMPLE_SAMPLE_ID)


@pytest.fixture
def allgraphs(base_session, example_sample):

    if not isfile(filepath):
        with base_session.with_cache(timeout=TIMEOUT) as sess:
            with timeit("Building new test graph"):
                adapter = AquariumAdapter(sess)

                with timeit("Initializing blueprint"):
                    blueprint = OperationBlueprintBuilder(AquariumAdapter(sess))

                with timeit("Collecting all deployed io values"):
                    blueprint.collect_deployed()

                with timeit("Collecting plans"):
                    plans = adapter.session.Plan.last(500)
                    blueprint.collect(plans)

                with timeit("Building blueprint graph"):
                    blueprint_graph = blueprint.build()

                with timeit("Building sample graph"):
                    s = example_sample
                    with base_session.with_cache(timeout=60) as sess:
                        adapter = AquariumAdapter(sess)
                        sample_graph = adapter.build_sample_graph([s])

                with timeit("Building operation graph"):
                    builder = OperationGraphBuilder(
                        AquariumAdapter(sess), blueprint_graph, sample_graph
                    )

                with timeit("Building anonymous graph"):
                    graph = builder.build_anon_graph()
        sample_graph.write(samplegraph_filepath)
        # sample_graph.write_gexf(samplegraph_filepath)

        graph.write(filepath)
        # graph.write_gexf(filepath)

        blueprint_graph.write(blueprintgraph_filepath)
        # blueprint_graph.write_gexf(blueprintgraph_filepath)
    else:
        sample_graph = SampleGraph.read(samplegraph_filepath)
        blueprint_graph = AFTGraph.read(blueprintgraph_filepath)
        graph = OperationGraph.read(filepath)
        builder = OperationGraphBuilder(
            AquariumAdapter(base_session.with_cache(timeout=60)),
            blueprint_graph,
            sample_graph,
        )
    return {
        "sample": sample_graph,
        "blueprint": blueprint_graph,
        "graph": graph,
        "builder": builder,
    }


@pytest.fixture
def graph(allgraphs):
    return allgraphs["graph"]


@pytest.fixture
def builder(allgraphs):
    return allgraphs["builder"]


@pytest.fixture
def sample_graph(allgraphs):
    return allgraphs["sample"]


@pytest.fixture
def blueprint(allgraphs):
    return allgraphs["blueprint"]
