import pytest
from terrarium import (
    Serializer,
    SampleGraphBuilder,
    ProtocolBlueprintBuilder,
    ProtocolGraphBuilder,
)
from more_itertools import spy


@pytest.mark.parametrize("num_plans", [1, 30])
def test_build(session):
    s = session.Sample.find(27608)
    sample_graph = SampleGraphBuilder.build([s])

    with session.with_cache(timeout=60) as sess:
        plans = sess.Plan.last(2)
        nodes, edges = Serializer.serialize_plans(plans)

    with session.with_cache(timeout=60) as sess:
        all_afts = Serializer.serialize_all_afts(sess)
    blueprint = ProtocolBlueprintBuilder().build(
        all_nodes=all_afts, nodes=nodes, edges=edges
    )

    graph = ProtocolGraphBuilder.build_graph(blueprint, sample_graph)

    print(spy(graph.nodes))
    print(spy(graph.edges))
