from terrarium.schemas.validate import validate_with_schema
from terrarium.utils import group_by, graph_utils
from terrarium.graphs import OperationGraph
from typing import Sequence
from itertools import chain
import networkx as nx
from itertools import product
import math


def extract_end_nodes(
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

    end_nodes = []
    for n, aft in graph.model_data("AllowableFieldType"):
        if validate_with_schema(aft, match):
            end_nodes.append(n)
    return end_nodes


# TODO: some operations are turning into roots even if they have inputs. Why???
def extract_root_operations(graph: OperationGraph) -> Sequence[dict]:
    """
    Extracts operations that have no inputs (such as "Order Primer")
    :param graph:
    :type graph:
    :return:
    :rtype:
    """
    is_output = lambda x: x["field_type"]["role"] == "output"
    nodes = [n for n, _ in graph.model_data("AllowableFieldType", is_output)]
    roots = graph_utils.get_roots(graph.graph, nodes)
    return roots


def extract_root_items(graph: OperationGraph) -> Sequence[dict]:
    items = [n for n, ndata in graph.model_data("Item")]
    return items


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


def top_paths(G, include_nodes, weight_key):
    """


        :param G: a nx.DiGraph
        :param include_nodes: list of nodes to include in the path
        :param weight_key: weight key
        :return:
        """

    path = []
    cost = 0
    for start, end in windowed(include_nodes, 2):
        path += nx.dijkstra_path(G, start, end, weight=weight_key)
        cost += nx.dijkstra_path_length(G, start, end, weight=weight_key)
    return cost, list(unique_justseen(path))


from terrarium.utils.color_utils import cprint
from more_itertools import windowed, unique_justseen


class Algorithms(object):
    def metric_closure(self):
        pass

    def _optimize_get_seed_paths(self, G, start_nodes, end_nodes, output_node=None):
        paths = []
        for start, end in product(start_nodes, end_nodes):
            through_nodes = [start, end]
            if output_node:
                through_nodes.append(output_node)
            try:
                cost, path = top_paths(G, through_nodes, "weight")
            except nx.exception.NetworkXNoPath:
                continue
            paths.append((cost, path))
        return paths

    def trim_paths(self):
        pass

    def optimize_steiner_set_tree(
        self,
        G,
        start_nodes,
        end_nodes,
        group_assignments,
        groups,
        visited,
        output_node,
        verbose=False,
    ):

        ############################################
        # 1. find all shortest paths
        ############################################

        non_visited_end_nodes = [e for e in end_nodes if e not in visited]
        seed_paths = self._optimize_get_seed_paths(
            G, start_nodes, non_visited_end_nodes, output_node
        )
        visited += end_nodes

        ############################################
        # 2. find overall shortest path(s)
        ############################################
        NUM_PATHS = 3
        THRESHOLD = 10 ** 8

        if not seed_paths:
            if verbose:
                cprint("No paths found", "red")
            return math.inf, [], visited
        seed_paths = sorted(seed_paths, key=lambda x: x[0])
        cost, path = seed_paths[0]
        final_paths = [path]
        if cost > THRESHOLD:
            cprint("Path beyond threshold, returning early", "red")
            cprint(graph_utils.get_path_length(G, path), "blue")
            return cost, final_paths, visited
        if verbose:
            cprint("Single path found with cost {}".format(cost), None, "blue")
            cprint(graph_utils.get_path_weights(G, path), None, "blue")

        ############################################
        # 3. mark edges as 'visited'
        ############################################
        graph_copy = G.copy()
        if len(path) > 1:
            for e1, e2 in windowed(path, 2):
                edge = graph_copy.edges[e1, e2]
                edge["weight"] = 0

        ############################################
        # 4. find steiner sets
        ############################################

        groups = [group_assignments.get(n, []) for n in path]
        for group in groups:
            pass

        # ############################################
        # # 4.1 input-to-output graph
        # ############################################
        # input_to_output = OrderedDict()
        # for n1, n2 in zip(path[:-1], path[1:]):
        #     node2 = graph_copy.get_node(n2)
        #     node1 = graph_copy.get_node(n1)
        #     if "sample" in node1:
        #         visited.add(node1["sample"].id)
        #     if "sample" in node2:
        #         visited.add(node2["sample"].id)
        #     if node2["node_class"] == "AllowableFieldType":
        #         aft2 = node2["model"]
        #         if aft2.field_type.role == "output":
        #             input_to_output[n1] = n2

        ############################################
        # 4.2  search for all unassigned inputs
        ############################################
        # iterate through each input to find unfullfilled inputs

        # inputs = list(input_to_output.keys())[:]
        # print(input_to_output.keys())
        # if depth > 0:
        #     inputs = inputs[:-1]
        #     print()
        #     print("INPUTS: {}".format(inputs))

        #     all_sister
        # empty_assignments = defaultdict(list)
        #
        # for i, n in enumerate(inputs):
        #     print()
        #     print("Finding sisters for:")
        #     self.print_aft(bgraph, n)
        #     output_n = input_to_output[n]
        #     ndata = bgraph_copy.get_node(n)
        #     sisters = self.get_sister_inputs(
        #         n, ndata, output_n, bgraph_copy, ignore=visited_end_nodes
        #     )
        #     if not sisters:
        #         print("no sisters found")
        #     for ftid, nodes in sisters.items():
        #         print("**Sister FieldType {}**".format(ftid))
        #         for s, values in nodes:
        #             self.print_aft(bgraph, s)
        #             empty_assignments["{}_{}".format(output_n, ftid)].append(
        #                 (s, output_n, values)
        #             )
        #         print()

        # TODO: Algorithm gets stuck on shortest top path...
        # e.g. Yeast Glycerol Stock to Yeast Mating instead of yeast transformation

        # 1. find all shortest paths
        # self.get_shortest_paths()

        # 2. find overall shortest path(s)
        # self.trim_paths()

        # 3. mark edges as 'visited'
        # 4.1 input-to-output graph
        # 4.2 search for all unassigned inputs

        ############################################
        # 4.3 recursively find cost & shortest paths
        #     for unassigned inputs for every possible
        #     assignment
        ############################################

        ############################################
        # 5 Make a sample penalty for missing input samples
        ############################################

        ############################################
        # 6 return cost and paths
        ############################################

    # if visited_samples is None:
    #     visited_samples = set()
    #
    #     ############################################
    #     # 1. find all shortest paths
    #     ############################################
    # seed_paths = self._optimize_get_seed_paths(
    #     start_nodes, end_nodes, bgraph, visited_end_nodes, output_node, verbose
    # )
    # visited_end_nodes += end_nodes
    #
    # ############################################
    # # 2. find overall shortest path(s)
    # ############################################
    # NUM_PATHS = 3
    # THRESHOLD = 10 ** 8
    #
    # if not seed_paths:
    #     if verbose:
    #         print("No paths found")
    #     return math.inf, [], visited_samples
    # seed_paths = sorted(seed_paths, key=lambda x: x[0])
    # cost, path = seed_paths[0]
    # final_paths = [path]
    # if cost > THRESHOLD:
    #     cprint("Path beyond threshold, returning early", "red")
    #     print(graph_utils.get_path_length(bgraph, path))
    #     return cost, final_paths, visited_samples
    #
    # if verbose:
    #     cprint("Single path found with cost {}".format(cost), None, "blue")
    #     cprint(graph_utils.get_path_weights(bgraph, path), None, "blue")
    #
    # ############################################
    # # 3. mark edges as 'visited'
    # ############################################
    # bgraph_copy = bgraph.copy()
    # edges = list(zip(path[:-1], path[1:]))
    # for e1, e2 in edges:
    #     edge = bgraph_copy.get_edge(e1, e2)
    #     edge["weight"] = 0
    #
    # ############################################
    # # 4.1 input-to-output graph
    # ############################################
    # input_to_output = OrderedDict()
    # for n1, n2 in zip(path[:-1], path[1:]):
    #     node2 = bgraph_copy.get_node(n2)
    #     node1 = bgraph_copy.get_node(n1)
    #     if "sample" in node1:
    #         visited_samples.add(node1["sample"].id)
    #     if "sample" in node2:
    #         visited_samples.add(node2["sample"].id)
    #     if node2["node_class"] == "AllowableFieldType":
    #         aft2 = node2["model"]
    #         if aft2.field_type.role == "output":
    #             input_to_output[n1] = n2
    #
    # ############################################
    # # 4.2  search for all unassigned inputs
    # ############################################
    # print("PATH:")
    # for p in path:
    #     print(p)
    #     self.print_aft(bgraph, p)
    #
    # # iterate through each input to find unfullfilled inputs
    # inputs = list(input_to_output.keys())[:]
    # print(input_to_output.keys())
    # if depth > 0:
    #     inputs = inputs[:-1]
    # #     print()
    # #     print("INPUTS: {}".format(inputs))
    #
    # #     all_sister
    # empty_assignments = defaultdict(list)
    #
    # for i, n in enumerate(inputs):
    #     print()
    #     print("Finding sisters for:")
    #     self.print_aft(bgraph, n)
    #     output_n = input_to_output[n]
    #     ndata = bgraph_copy.get_node(n)
    #     sisters = self.get_sister_inputs(
    #         n, ndata, output_n, bgraph_copy, ignore=visited_end_nodes
    #     )
    #     if not sisters:
    #         print("no sisters found")
    #     for ftid, nodes in sisters.items():
    #         print("**Sister FieldType {}**".format(ftid))
    #         for s, values in nodes:
    #             self.print_aft(bgraph, s)
    #             empty_assignments["{}_{}".format(output_n, ftid)].append(
    #                 (s, output_n, values)
    #             )
    #         print()
    #
    # ############################################
    # # 4.3 recursively find cost & shortest paths
    # #     for unassigned inputs for every possible
    # #     assignment
    # ############################################
    # all_assignments = list(product(*empty_assignments.values()))
    # print(all_assignments)
    # for k, v in empty_assignments.items():
    #     print(k)
    #     print(v)
    # if all_assignments[0]:
    #
    #     # TODO: enforce unique sample_ids if in operation_type
    #     cprint("Found {} assignments".format(len(all_assignments)), None, "blue")
    #     best_assignment_costs = []
    #
    #     for assign_num, assignment in enumerate(all_assignments):
    #         cprint(
    #             "Evaluating assignment {}/{}".format(
    #                 assign_num + 1, len(all_assignments)
    #             ),
    #             None,
    #             "red",
    #         )
    #         cprint("Assignment length: {}".format(len(assignment)), None, "yellow")
    #
    #         assignment_cost = 0
    #         assignment_paths = []
    #         assignment_samples = set(visited_samples)
    #         for end_node, output_node, _ in assignment:
    #             _cost, _paths, _visited_samples = self.optimize_steiner_tree(
    #                 start_nodes,
    #                 [end_node],
    #                 bgraph_copy,
    #                 visited_end_nodes[:],
    #                 assignment_samples,
    #                 output_node,
    #                 verbose=True,
    #                 depth=depth + 1,
    #             )
    #             assignment_cost += _cost
    #             assignment_paths += _paths
    #             assignment_samples = assignment_samples.union(_visited_samples)
    #         best_assignment_costs.append(
    #             (assignment_cost, assignment_paths, assignment_samples)
    #         )
    #     cprint([(len(x[2]), x[0]) for x in best_assignment_costs], "green")
    #     best_assignment_costs = sorted(
    #         best_assignment_costs, key=lambda x: (-len(x[2]), x[0])
    #     )
    #
    #     cprint(
    #         "Best assignment cost returned: {}".format(best_assignment_costs[0][0]),
    #         "red",
    #     )
    #
    #     cost += best_assignment_costs[0][0]
    #     final_paths += best_assignment_costs[0][1]
    #     visited_samples = visited_samples.union(best_assignment_costs[0][2])
    #
    # ############################################
    # # 5 Make a sample penalty for missing input samples
    # ############################################
    #
    # output_samples = set()
    # for path in final_paths:
    #     for node in path:
    #         ndata = bgraph_copy.get_node(node)
    #         if "sample" in ndata:
    #             output_samples.add(ndata["sample"])
    #
    # expected_samples = set()
    # for sample in output_samples:
    #     for pred in self.sample_composition.predecessors(sample.id):
    #         expected_samples.add(pred)
    #
    # ############################################
    # # return cost and paths
    # ############################################
    #
    # sample_penalty = max(
    #     [(len(expected_samples) - len(visited_samples)) * 10000, 0]
    # )
    # cprint("SAMPLES {}/{}".format(len(visited_samples), len(expected_samples)))
    # cprint("COST AT DEPTH {}: {}".format(depth, cost), None, "red")
    # cprint("SAMPLE PENALTY: {}".format(sample_penalty))
    # cprint("VISITED SAMPLES: {}".format(visited_samples), None, "red")
    # return cost + sample_penalty, final_paths, visited_samples
