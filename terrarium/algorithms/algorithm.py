import networkx as nx
from itertools import product
import math
from terrarium import constants as C
from terrarium.utils import graph_utils
from termcolor import cprint
from more_itertools import windowed


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
                cost, path = graph_utils.top_paths(G, through_nodes, "weight")
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
            cprint(graph_utils.get_path_length(G, path, C.WEIGHT), "blue")
            return cost, final_paths, visited
        if verbose:
            cprint("Single path found with cost {}".format(cost), "blue")
            cprint(graph_utils.get_path_weights(G, path, C.WEIGHT), "blue")

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
