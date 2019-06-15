from terrarium import SampleGraphBuilder


def test_build_sample_graph(session):
    s = session.Sample.find(27608)
    g = SampleGraphBuilder.build([s])

    assert len(list(g.nodes)) > 1
    assert len(list(g.edges)) > 1
