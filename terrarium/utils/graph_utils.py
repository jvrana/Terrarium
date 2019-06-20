import networkx as nx


def get_roots(G, nbunch=None):
    return [n for (n, d) in G.in_degree() if d == 0 and (n in nbunch or nbunch is None)]


def get_leaves(G, nbunch=None):
    return [
        n for (n, d) in G.out_degree() if d == 0 and (n in nbunch or nbunch is None)
    ]


def get_edges_from_path(path):
    return zip(path[:-1], path[1:])


def get_path_weights(graph, path, weight):
    edge_weights = []
    for e1, e2 in get_edges_from_path(path):
        edge = graph.edges[e1, e2]
        edge_weights.append(edge[weight])
    return edge_weights


def get_path_length(graph, path, weight):
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
