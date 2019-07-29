from terrarium import Terrarium

from terrarium.adapters import AquariumAdapter
from terrarium import OperationBlueprintBuilder, OperationGraphBuilder


def test_builder_api(base_session, example_sample):
    """Test the complete builder API here"""

    with base_session.with_cache(timeout=60) as sess:
        adapter = AquariumAdapter(sess)
        blueprint = OperationBlueprintBuilder(adapter)
        blueprint.collect_deployed()
        blueprint.collect(sess.Plan.last(10))
        blueprint_graph = blueprint.build()
        sample_graph = adapter.build_sample_graph([example_sample])
        builder = OperationGraphBuilder(adapter, blueprint_graph, sample_graph)
        graph = builder.build_anon_graph()


def test_terrarium(base_session):
    with base_session.with_cache(timeout=60) as sess:
        adapter = AquariumAdapter(sess)
        terra = Terrarium(adapter)
        print(adapter)
        print(terra.info())
