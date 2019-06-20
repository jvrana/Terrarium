from terrarium.algorithms import (
    Algorithms,
    input_groups,
    top_paths,
    extract_root_operations,
    iter_root_items,
    iter_end_nodes,
)

import networkx as nx
import pytest


def test_main(graph):
    pass


def test_algorithm(base_session, graph, example_sample):
    A = Algorithms()

    # cleaned = clean_graph(graph.copy())
    # G = cleaned.graph
    G = graph.graph

    start_item_nodes = iter_root_items(graph)
    start_op_nodes = extract_root_operations(graph)
    start_nodes = start_item_nodes + start_op_nodes

    with base_session() as sess:
        object_type = sess.ObjectType.find_by_name("Yeast Glycerol Stock")

    end_nodes = iter_end_nodes(
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
