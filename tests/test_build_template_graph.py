from terrarium import TemplateGraphBuilder, Serializer


def test_build_template_graph(session):

    with session.with_cache(timeout=60) as sess:
        plans = sess.Plan.last(30)
        nodes, edges = Serializer.serialize_plans(plans)

    with session.with_cache(timeout=60) as sess:
        all_afts = Serializer.serialize_all_afts(sess)

    graph_builder = TemplateGraphBuilder()
    graph_builder.build_template_graph(all_afts, nodes, edges)
