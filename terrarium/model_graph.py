import networkx as nx


class ModelGraph(object):
    def __init__(self):
        self.graph = nx.DiGraph()
        self.prefix = ''
        self.suffix = ''

    @property
    def nodes(self):
        """
        Return node iterator

        :return:
        :rtype:
        """
        return self.graph.nodes

    @property
    def edges(self):
        """
        Return edges iterator.

        :return:
        :rtype:
        """
        return self.graph.edges

    def node_id(self, model):
        s = "{}_{}".format(model["__class__"], model["primary_key"])
        return self.prefix + s + self.suffix

    def set_prefix(self, prefix):
        """Sets this graph to have a prefix."""
        mapping = {n: prefix + n for n in self.nodes}
        self.graph = nx.relabel_nodes(self.graph, mapping)
        self.prefix = prefix

    def set_suffix(self, suffix):
        mapping = {n: n + suffix for n in self.nodes}
        self.graph = nx.relabel_nodes(self.graph, mapping)
        self.suffix = suffix

    def add_model(self, model_data):
        self.graph.add_node(self.node_id(model_data), data=model_data)

    def add_edge(self, n1, n2, **kwargs):
        assert n1 in self.graph
        assert n2 in self.graph
        self.graph.add_edge(n1, n2, **kwargs)

    def add_edge_from_models(self, m1, m2, **kwargs):
        n1 = self.node_id(m1)
        n2 = self.node_id(m2)
        if n1 not in self.graph:
            self.add_model(m1)
        if n2 not in self.graph:
            self.add_model(m2)
        self.graph.add_edge(n1, n2, **kwargs)

    @staticmethod
    def copy_graph(graph):
        graph_copy = nx.DiGraph()
        graph_copy.add_nodes_from(graph.nodes(data=True))
        graph_copy.add_edges_from(graph.edges(data=True))
        return graph_copy

    def _resolve_item(self, item):
        if isinstance(item, int) or isinstance(item, str):
            return item
        elif issubclass(type(item), dict):
            return self.node_id(item)
        else:
            return None

    def _array_to_identifiers(self, nodes):
        formatted_nodes = []
        for n in nodes:
            node_id = self._resolve_item(n)
            if node_id:
                formatted_nodes.append(node_id)
            else:
                raise TypeError(
                    "Type '{}' {} not recognized as a node".format(type(n), n)
                )
        return formatted_nodes
git
    def subgraph(self, nodes):
        nodes = self._array_to_identifiers(nodes)

        graph_copy = nx.DiGraph()
        graph_copy.add_nodes_from((n, self.nodes[n]) for n in nodes)

        edges = []
        for n1, n2 in self.edges:
            if n1 in nodes and n2 in nodes:
                edges.append((n1, n2, self.edges[n1, n2]))

        graph_copy.add_edges_from(edges)

        browser_graph_copy = self.copy()
        browser_graph_copy.graph = graph_copy
        return browser_graph_copy

    def copy(self):
        return self.__copy__()

    def __contains__(self, item):
        return ite

    def __copy__(self):
        copied = self.__class__()
        copied.graph = self.copy_graph(self.graph)
        return copied


def samples_to_parent_graphs(sample, g=None, visited=[]):
    if g is None:
        g = nx.DiGraph()
    if node_id(sample) in visited:
        return g
    else:
        visited.append(g.node_id(sample))

    g.add_node("Sample_{}".format(sample._primary_key))

    fvs = sample.field_values

    for fv in fvs:
        if fv.sample:
            g.add_edge_from_models(sample, fv.sample)
            samples_to_parent_graphs(fv.sample, g=g, visited=visited)
    return g
