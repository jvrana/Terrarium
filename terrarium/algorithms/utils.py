from terrarium.schemas.validate import validate_with_schema
from terrarium.utils import group_by
from terrarium.graphs import OperationGraph
from typing import Sequence
from itertools import chain


def iter_end_nodes(
    graph: OperationGraph, goal_sample_id: int, goal_object_type_id: int
) -> Sequence[dict]:
    """
    Extract any afts that match the goal_sample_id and goal_object_type_id

    :param graph:
    :param goal_sample_id:
    :param goal_object_type_id:
    :return:
    """

    match = {
        "sample": {"id": goal_sample_id},
        "object_type_id": goal_object_type_id,
        "field_type": {"role": "output"},
    }

    for n, aft in graph.model_data("AllowableFieldType"):
        if validate_with_schema(aft, match):
            yield n, aft


# TODO: some operations are turning into roots even if they have inputs. Why???
def iter_root_operations(graph: OperationGraph) -> Sequence[dict]:
    """
    Extracts operations that have no inputs (such as "Order Primer")

    :param graph:
    :type graph:
    :return:
    :rtype:
    """
    for n, ndata in graph.model_data("AllowableFieldType"):
        fts = ndata["field_type"]["operation_type"]["field_types"]
        inputs = [ft for ft in fts if ft["role"] == "input"]
        if not inputs:
            yield n, ndata


def iter_root_items(graph: OperationGraph) -> Sequence[dict]:
    return graph.model_data("Item")


def input_groups(graph: OperationGraph) -> Sequence[str]:
    """
    Assigns input nodes to a group. Returns a dictionary
    of nodes ids to group indices `{node_id: group_index}` and
    an ordered list of grouped_ids `[(node_id1, node_id2, ...), (...), ...]`

    :param graph:
    :return:
    """
    is_input = lambda x: x["field_type"]["role"] == "input"
    is_array = lambda x: bool(x["field_type"]["array"])
    node_arrays = graph.model_data(
        "AllowableFieldType", [is_input, lambda x: ~is_array(x)]
    )
    nodes = graph.model_data("AllowableFieldType", [is_input, lambda x: ~is_array(x)])

    def hasher(*args):
        return "__".join([str(s) for s in args])

    def node_hasher(x):
        return hasher(x[1]["field_type_id"], x[1]["field_type"]["parent_id"])

    def node_array_hasher(x):
        return hasher(node_hasher(x), x[1]["sample"]["id"])

    node_groups = group_by(nodes, node_hasher)
    node_array_groups = group_by(node_arrays, node_array_hasher)

    groups = chain(node_groups.values(), node_array_groups.values())

    group_dict = {}
    group_list = []

    for group_index, group in enumerate(groups):
        node_ids = tuple([n for n, _ in group])
        group_list.append(node_ids)
        for n, ndata in group:
            group_dict[n] = group_index
    return group_dict, group_list
