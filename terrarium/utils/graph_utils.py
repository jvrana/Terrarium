###################################################
#
# Networkx utilities
#
###################################################
import networkx as nx


def get_edges_from_path(path):
    return zip(path[:-1], path[1:])


def get_path_weights(graph, path, weight="weight"):
    edge_weights = []
    for e1, e2 in get_edges_from_path(path):
        edge = graph.edges[e1, e2]
        edge_weights.append(edge[weight])
    return edge_weights


def get_path_length(graph, path, weight="weight"):
    length = 0
    for e1, e2 in get_edges_from_path(path):
        edge = graph.edges[e1, e2]
        length += edge[weight]
    return length


def iter_top_paths(
    graph, bellman_ford_path_length_dict, start_nodes, end_nodes, cutoff
):
    for start in start_nodes:
        for end in end_nodes:
            shortest_length = bellman_ford_path_length_dict[start].get(end, None)
            if shortest_length is not None and shortest_length <= cutoff:
                for path in nx.all_shortest_paths(graph, start, end):
                    path_length = get_path_length(graph, path)
                    if path_length <= cutoff:
                        yield path_length, path


def top_paths(nodes, graph, weight="weight"):
    """Find min path through a list of nodes."""

    all_paths = []
    total_cost = 0
    for start, end in zip(nodes[:-1], nodes[1:]):
        path = nx.dijkstra_path(graph.graph, start, end, weight=weight)
        cost = nx.dijkstra_path_length(graph.graph, start, end, weight=weight)
        all_paths.append(path)
        total_cost += cost
    if len(all_paths) == 1:
        return (total_cost, all_paths[0])
    else:
        total_path = []
        for path in all_paths[:-1]:
            total_path += path[:-1]
        total_path += all_paths[-1]
        return (total_cost, total_path)


def find_leaves(graph):
    leaves = []
    for n in graph:
        if not len(list(graph.successors(n))):
            leaves.append(n)
    return leaves


def find_roots(graph):
    roots = []
    for n in graph:
        if not len(list(graph.predecessors(n))):
            roots.append(n)
    return roots


# def _topological_sort_helper(graph):
#     """Attempt a rudimentary topological sort on the plan"""
#
#     _x, _y = self.TOP_RIGHT
#
#     y = _y
#     delta_x = self.BOX_DELTA_X
#     delta_y = -self.BOX_DELTA_Y
#
#     max_depth = {}
#     roots = self.roots()
#     for root in roots:
#         depths = nx.single_source_shortest_path_length(self.G, root)
#         for n, d in depths.items():
#             max_depth[n] = max(max_depth.get(n, d), d)
#
#     # push roots 'up' so they are not stuck on layer one
#     for root in self.roots():
#         successors = list(self.successors(root))
#         if len(successors) > 0:
#             min_depth = min([max_depth[s] for s in successors])
#             max_depth[root] = min_depth - 1
#
#     by_depth = OrderedDict()
#     for node, depth in max_depth.items():
#         by_depth.setdefault(depth, [])
#         by_depth[depth].append(node)
#     for depth in sorted(by_depth):
#         op_ids = by_depth[depth]
#
#         # sort by predecessor_layout
#         predecessor_avg_x = []
#         x = 0
#         for op_id in op_ids:
#             predecessors = list(self.predecessors(op_id))
#             if len(predecessors) > 0:
#                 x, _ = self.subgraph(predecessors).midpoint()
#                 predecessor_avg_x.append((x, op_id))
#             else:
#                 predecessor_avg_x.append((x + 1, op_id))
#         sorted_op_ids = [op_id for _, op_id in sorted(
#             predecessor_avg_x, key=lambda x: x[0])]
#
#         x = _x
#         # sorted_op_ids = sorted(op_ids)
#         ops = self.nodes_to_ops(sorted_op_ids)
#         for op in ops:
#             op.x = x
#             op.y = y
#             x += delta_x
#         layer = self.subgraph(op_ids)
#         predecessor_layout = self.predecessor_layout(layer)
#         layer.align_x_midpoints_to(predecessor_layout)
#         y += delta_y
#
#     # readjust
#     self.move(_x, _y)
