"""
Fixtures for algorithm tests.

Produces a graphs in './data' as .gexf files if None exists. Otherwise loads the gexf files for other tests.
"""

from os.path import dirname, join, abspath, isfile, isdir
from os import makedirs
import pytest
from terrarium.graphs import OperationGraph
from terrarium.builders import OperationBlueprintBuilder, OperationGraphBuilder
from terrarium.adapters import AquariumAdapter
from time import time

EXAMPLE_SAMPLE_ID = 27608
TIMEOUT = 60
NUM_PLANS = 30
FILENAME = "operation_graph.json"
DIRNAME = "data"

data_path = join(dirname(abspath(__file__)), DIRNAME)
if not isdir(data_path):
    makedirs(data_path)
filepath = join(data_path, FILENAME)


EXAMPLE_SAMPLE_ID = 27608


class timeit(object):
    def __init__(self, name):
        self.name = name
        self.t1 = None

    def __enter__(self):
        print("Started '{}'".format(self.name))
        self.name
        self.t1 = time()

    def __exit__(self, a, b, c):
        delta = time() - self.t1
        print("Finished '{}' in {} seconds".format(self.name, delta))


@pytest.fixture
def example_sample(base_session):
    return base_session.Sample.find(EXAMPLE_SAMPLE_ID)


@pytest.fixture
def graph(base_session, example_sample):

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
        sample_graph.write_gexf(join(data_path, "sample_graph.gexf"))

        graph.write(filepath)
        graph.write_gexf(join(data_path, "operation_graph.gexf"))

        blueprint_graph.write(join(data_path, "blueprint.json"))
        blueprint_graph.write_gexf(join(data_path, "blueprint.gexf"))
    else:
        graph = OperationGraph.read(filepath)
    return graph
