from os.path import join


def test_write_gexf(base_session, datadir, graph, example_sample, tmp_path):
    filename = join(datadir, "operation_graph.gexf")
    graph.write_gexf(filename)
