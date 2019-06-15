from terrarium import ProtocolBlueprintBuilder, Serializer


def test_build_template_graph(session):

    with session.with_cache(timeout=60) as sess:
        plans = sess.Plan.last(30)
        nodes, edges = Serializer.serialize_plans(plans)

    with session.with_cache(timeout=60) as sess:
        all_afts = Serializer.serialize_all_afts(sess)

    blueprint = ProtocolBlueprintBuilder()
    blueprint.build(all_afts, nodes, edges)
