from terrarium.algorithms import (
    Algorithms,
    input_groups,
    top_paths,
    extract_root_operations,
    extract_root_items,
    extract_end_nodes,
)

import networkx as nx
import pytest


def test_main(graph):
    pass


class TestGraphUtils(object):
    def test_input_groups(self, graph):
        group_dict, groups = input_groups(graph)
        for k, v in group_dict.items():
            assert k in groups[v]

        for group in groups:
            for n in group:
                assert n in group_dict

    def test_top_paths(self):
        G = nx.path_graph(5, create_using=nx.DiGraph)
        assert top_paths(G, [0, 4], None)
        assert top_paths(G, [0, 1, 4], None)
        with pytest.raises(nx.exception.NetworkXNoPath):
            assert not top_paths(G, [1, 0, 4], None)
            assert not top_paths(G, [1, 5], None)

    def test_extract_root_operations(self, graph):
        nodes = extract_root_operations(graph)

    def test_extract_items(self, graph):
        extract_root_items(graph)

    def test_extract_end_nodes(self, session, graph):
        extract_end_nodes(graph, 1, 1)


def test_algorithm(base_session, graph, example_sample):
    A = Algorithms()

    # cleaned = clean_graph(graph.copy())
    # G = cleaned.graph
    G = graph.graph

    start_item_nodes = extract_root_items(graph)
    start_op_nodes = extract_root_operations(graph)
    start_nodes = start_item_nodes + start_op_nodes

    with base_session() as sess:
        object_type = sess.ObjectType.find_by_name("Yeast Glycerol Stock")

    end_nodes = extract_end_nodes(
        graph, goal_object_type_id=object_type.id, goal_sample_id=example_sample.id
    )
    group_assignments, groups = input_groups(graph)
    A.optimize_steiner_set_tree(
        G=G,
        start_nodes=start_nodes,
        end_nodes=end_nodes,
        group_assignments=group_assignments,
        groups=groups,
        visited=[],
        output_node=None,
        verbose=True,
    )
