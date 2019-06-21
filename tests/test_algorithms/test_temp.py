from os.path import dirname, abspath, join


def test_write_gexf(base_session, graph, example_sample, tmp_path):
    here = dirname(abspath(__file__))
    filename = join(here, "data", "operation_graph.gexf")
    graph.write_gexf(filename)
