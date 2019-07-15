import networkx as nx
from more_itertools import windowed, unique_justseen, pairwise
from typing import List, Union, Tuple


def get_roots(G: nx.Digraph, nbunch=None) -> List[Union[str, int]]:
    """Return the roots of the graph."""
    return [n for (n, d) in G.in_degree() if d == 0 and (n in nbunch or nbunch is None)]


def get_leaves(G: nx.DiGraph, nbunch=None) -> List[Union[str, int]]:
    """Returns the leaves of the graph"""
    return [
        n for (n, d) in G.out_degree() if d == 0 and (n in nbunch or nbunch is None)
    ]


def get_edges_from_path(path: List[Union[str, int]]):
    """Return pairwise edges of a path"""
    return pairwise(path)


def get_path_weights(
    graph: nx.DiGraph, path: List[Union[str, int]], weight: str
) -> List[float]:
    """Return list of edge weights in the path"""
    edge_weights = []
    for e1, e2 in get_edges_from_path(path):
        edge = graph.edges[e1, e2]
        edge_weights.append(edge[weight])
    return edge_weights


def get_path_length(
    graph: nx.DiGraph, path: List[Union[str, int]], weight: str
) -> float:
    """Return path length keyed by 'weight' key."""
    return sum(get_path_weights(graph, path))


def iter_top_paths(
    graph: nx.DiGraph,
    bellman_ford_path_length_dict: dict,
    start_nodes: List[Union[str, int]],
    end_nodes: List[Union[str, int]],
    cutoff: float,
):
    """
    Return the top paths between the start and end nodes
    :param graph:
    :param bellman_ford_path_length_dict:
    :param start_nodes:
    :param end_nodes:
    :param cutoff:
    :return:
    """
    for start in start_nodes:
        for end in end_nodes:
            shortest_length = bellman_ford_path_length_dict[start].get(end, None)
            if shortest_length is not None and shortest_length <= cutoff:
                for path in nx.all_shortest_paths(graph, start, end):
                    path_length = get_path_length(graph, path)
                    if path_length <= cutoff:
                        yield path_length, path


def top_paths(
    G: nx.Digraph, include_nodes: List[Union[str, int]], weight_key: str
) -> Tuple(float, List[Union[int, str]]):
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
