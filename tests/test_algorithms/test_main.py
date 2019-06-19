from terrarium.algorithms import (
    Algorithms,
    input_groups,
    top_paths,
    extract_root_operations,
    extract_root_items,
    clean_graph,
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

        for n in nodes:
            preds = list(graph.graph.predecessors(n))
            assert not preds

        parent_ids = []
        for node in nodes:
            ndata = graph.get_data(node)
            if ndata["__class__"] == "AllowableFieldType":
                parent_ids.append(ndata["field_type"]["parent_id"])

        with base_session.with_cache(timeout=60) as sess:
            operation_types = sess.OperationType.find(parent_ids)
            sess.browser.get("OperationType", "field_types")

        msgs = []
        for node in nodes:
            ndata = graph.get_data(node)
            if ndata["__class__"] == "AllowableFieldType":
                optype = sess.OperationType.find(ndata["field_type"]["parent_id"])
                inputs = [
                    ft
                    for ft in optype.field_types
                    if ft.ftype == "sample" and ft.role == "input"
                ]
                if inputs:

                    msgs.append(
                        "An root AFT {} was found that has inputs for '{}' {}".format(
                            node, optype.name, optype.deployed
                        )
                    )
                    for ft in inputs:
                        msgs.append("\t{} {} {}".format(ft.name, ft.array, ft.part))
                    print(list(graph.graph.predecessors(node)))
        raise Exception("\n".join(msgs))

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
