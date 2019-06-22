from terrarium.adapters import AquariumAdapter
from terrarium.builders import OperationBlueprintBuilder


def test_main(base_session, example_sample):

    s = example_sample
    with base_session.with_cache(timeout=60) as sess:
        adapter = AquariumAdapter(sess)
        sample_graph = adapter.build_sample_graph([s])

    with base_session.with_cache(timeout=60) as sess:
        adapter = AquariumAdapter(sess)
        blueprint = OperationBlueprintBuilder(adapter)
        blueprint.collect_deployed()
        blueprint.init_multidigraph()
        # plans = adapter.session.Plan.last(30)
        # blueprint.collect(plans)
        # return blueprint.build()
