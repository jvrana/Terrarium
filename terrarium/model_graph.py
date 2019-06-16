import networkx as nx
import json


class ModelGraph(object):

    PREFIX_KEY = 'prefix'
    SUFFIX_KEY = 'suffix'

    def __init__(self, name=None):
        self._graph = nx.DiGraph()
        self._graph.name = name
        self.prefix = ""
        self.suffix = ""

    @property
    def prefix(self):
        return self._graph.graph[self.PREFIX_KEY]

    @property
    def suffix(self):
        return self._graph.graph[self.SUFFIX_KEY]

    @prefix.setter
    def prefix(self, p):
        self._graph.graph[self.PREFIX_KEY] = p

    @suffix.setter
    def suffix(self, s):
        self._graph.graph[self.SUFFIX_KEY] = s

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, g):
        self._graph = g

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

    def add_data(self, model_data):
        self.graph.add_node(self.node_id(model_data), data=model_data)

    def get_data(self, node_id):
        return self.graph.nodes[node_id]["data"]

    def get_edge(self, n1, n2):
        return self.graph[n1][n2]

    def add_edge(self, n1, n2, **kwargs):
        assert n1 in self.graph
        assert n2 in self.graph
        self.graph.add_edge(n1, n2, **kwargs)

    def add_edge_from_models(self, m1, m2, **kwargs):
        n1 = self.node_id(m1)
        n2 = self.node_id(m2)
        if n1 not in self.graph:
            self.add_data(m1)
        if n2 not in self.graph:
            self.add_data(m2)
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

    def contains_model(self, model):
        return self.node_id(model) in self

    def json(self):
        return nx.adjacency_data(self.graph)

    @classmethod
    def load(cls, json_data):
        graph = nx.adjacency_graph(json_data)
        model_graph = cls()
        model_graph.graph = graph
        return model_graph

    def write(self, path):
        with open(path, 'w') as f:
            json.dump(self.json(), f)

    @classmethod
    def read(cls, path):
        with open(path, 'r') as f:
            cls.load(json.load(f))

    def save_gexf(self, path):
        return nx.write_gexf(self.graph, path)

    # TODO: put prefix and suffix on metadata attributes using XML?
    @classmethod
    def load_gexf(cls, path, prefix='', suffix=''):
        graph = nx.read_gexf(path)
        model_graph = cls()
        model_graph.graph = graph
        model_graph.prefix = prefix
        model_graph.suffix = suffix
        return model_graph

    def __len__(self):
        return self.graph.number_of_nodes()

    def __contains__(self, item):
        return item in self.graph

    def __copy__(self):
        copied = self.__class__()
        copied.graph = self.copy_graph(self.graph)
        return copied
