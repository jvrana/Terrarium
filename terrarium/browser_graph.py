import networkx as nx
from pydent.base import ModelBase as TridentBase
from typing import List


class BrowserGraph(object):
    """Graph class for representing Aquarium model-to-model relationships."""

    class DEFAULTS:

        MODEL_TYPE = "model"
        NODE_TYPE = "node"

    def __init__(self, browser):
        self.browser = browser
        self.graph = nx.DiGraph()
        self.model_hashes = {}
        self.prefix = ""
        self.suffix = ""

    def node_id(self, model):
        """
        Convert a pydent model into a unique graph id

        :param model: Trident model
        :type model: ModelBase
        :return: unique graph id for model
        :rtype: basestring
        """
        model_class = model.__class__.__name__
        model_hash = self.model_hashes.get(
            model_class,
            lambda model: "{cls}_{mid}".format(
                cls=model.__class__.__name__, mid=model.id
            ),
        )
        return self.prefix + model_hash(model) + self.suffix

    def set_prefix(self, prefix):
        mapping = {n: prefix + n for n in self.nodes}
        self.graph = nx.relabel_nodes(self.graph, mapping)
        self.prefix = prefix

    def set_suffix(self, suffix):
        mapping = {n: n + suffix for n in self.nodes}
        self.graph = nx.relabel_nodes(self.graph, mapping)
        self.suffix = suffix

    # def add_special_node(self, node_id, node_class):
    #     self.graph.add_node(node_id, node_class=node_class, model_id=None)
    #
    # def get_special_node(self, node_id):
    #     return self.graph.node(node_id)

    def add_special_node(self, node_id, node_class):
        return self.graph.add_node(
            node_id, node_class=node_class, node_type=self.DEFAULTS.NODE_TYPE
        )

    def add_model(self, model, node_id=None):
        """
        Add a model node to the graph with optional node_id

        :param model: Trident model
        :type model: ModelBase
        :param node_id: optional node_id to use
        :type node_id: basestring
        :return: None
        :rtype: None
        """
        model_class = model.__class__.__name__
        if not issubclass(type(model), TridentBase):
            raise TypeError(
                "Add node expects a Trident model, not a {}".format(type(model))
            )
        if node_id is None:
            node_id = self.node_id(model)
        return self.graph.add_node(
            node_id,
            node_class=model_class,
            model_id=model.id,
            node_type=self.DEFAULTS.MODEL_TYPE,
        )

    def add_edge_from_models(self, m1, m2, edge_type=None, **kwargs):
        """
        Adds an edge from two models.

        :param m1:
        :type m1:
        :param m2:
        :type m2:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        self.add_edge(
            self.node_id(m1),
            self.node_id(m2),
            model1=m1,
            model2=m2,
            edge_type=edge_type,
            **kwargs
        )

    def add_edge(self, n1, n2, model1=None, model2=None, edge_type=None, **kwargs):
        """
        Adds edge between two nodes given the node ids. Raises error if node does not exist
        and models are not provided. If node_id does not exist and model is provided, a new
        node is added.

        :param n1: first node id
        :type n1: int
        :param n2: second node id
        :type n2: int
        :param model1: first model (optional)
        :type model1: ModelBase
        :param model2: second model (optional)
        :type model2: ModelBase
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        if n1 not in self.graph:
            if model1 is not None:
                self.add_model(model1, n1)
            else:
                raise ValueError("Model1 must be provided")

        if n2 not in self.graph:
            if model1 is not None:
                self.add_model(model2, n2)
            else:
                raise ValueError("Model2 must be provided")

        return self.graph.add_edge(n1, n2, edge_type=edge_type, **kwargs)

    def update_node(self, node_id, data):
        node = self.get_node(node_id)
        node.update(data)
        return node

    def predecessors(self, node_id):
        return self.graph.predecessors(node_id)

    def successors(self, node_id):
        return self.graph.successors(node_id)

    @classmethod
    def _convert_id(cls, n):
        if isinstance(n, int) or isinstance(n, str):
            return n
        elif issubclass(type(n), TridentBase):
            return cls.node_id(n)
        else:
            raise TypeError("Type '{}' {} not recognized as a node".format(type(n), n))

    def get_node(self, node_id):
        """
        Get a node from a node_id. If provided with Trident model, model is converted into
        a node_id.
        :param node_id:
        :type node_id:
        :return:
        :rtype:
        """

        node = self.graph.node[node_id]
        if "model_id" in node and "model" not in node:
            model = self.browser.find(node["model_id"], model_class=node["node_class"])
            node["model"] = model
        return node

    def get_model(self, node_id):
        node = self.get_node(node_id)
        return node.get("model", None)

    def get_edge(self, n1, n2):
        return self.graph.edges[n1, n2]

    def models(self, model_class=None):
        return list(self.iter_models(model_class))

    @property
    def nodes(self):
        return self.graph.nodes

    @property
    def edges(self):
        return self.graph.edges

    def iter_node_data(self, node_type=None, nbunch=None):
        if nbunch is None:
            nbunch = self
        for n in nbunch:
            node = self.get_node(n)
            if node_type is None or node["node_type"] == node_type:
                yield (n, node)

    def iter_edge_data(self, ebunch=None):
        if ebunch is None:
            ebunch = self
        for e in ebunch:
            yield (e, self.get_edge(*e))

    def iter_model_data(self, model_class=None, nbunch=None, **attrs):
        for n, ndata in self.iter_node_data(
            node_type=self.DEFAULTS.MODEL_TYPE, nbunch=nbunch
        ):
            if model_class is None or ndata["node_class"] == model_class:
                model = ndata["model"]
                passes = True
                for attr, val in attrs.items():
                    if not hasattr(model, attr) or getattr(model, attr) != val:
                        passes = False
                        break
                if passes:
                    yield (n, ndata)

    def iter_models(self, model_class=None, nbunch=None, **attrs):
        for n, ndata in self.iter_model_data(model_class, nbunch=nbunch, **attrs):
            yield ndata["model"]

    @staticmethod
    def copy_graph(graph):
        graph_copy = nx.DiGraph()
        graph_copy.add_nodes_from(graph.nodes(data=True))
        graph_copy.add_edges_from(graph.edges(data=True))
        return graph_copy

    @classmethod
    def _array_to_identifiers(cls, nodes):
        formatted_nodes = []
        for n in nodes:
            if isinstance(n, int) or isinstance(n, str):
                formatted_nodes.append(n)
            elif issubclass(type(n), TridentBase):
                formatted_nodes.append(cls.node_id(n))
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

    def filter(self, key=None):
        if key is None:
            key = lambda x: True
        nodes = [n for n in self.nodes if key(n)]
        return self.subgraph(nodes)

    def remove(self, key=None):
        if key is None:
            key = lambda x: False
        nodes = [n for n in self.nodes if key(n)]
        return self.subgraph(set(self.graph.nodes).difference(set(nodes)))

    def only_models(self, model_class=None, **attrs):
        return self.subgraph(
            [n for n, _ in self.iter_model_data(model_class=model_class, **attrs)]
        )

    def select_nodes(self, model_class=None, key=None):
        selected = set()
        if key:
            for n, ndata in self.iter_model_data(model_class=model_class):
                if key(ndata["model"]):
                    selected.add(n)
        return selected

    def difference(self, node_list: List[str]):
        node_set = set(self.nodes)
        return self.subgraph(node_set.difference(node_list))

    def cache_models(self):
        models = {}
        for n in self:
            ndata = self.graph.node[n]
            if "model_id" in ndata:
                models.setdefault(ndata["node_class"], []).append(ndata["model_id"])

        models_by_id = {}
        for model_class, model_ids in models.items():
            found_models = self.browser.find(model_ids, model_class=model_class)
            models_by_id.update({m.id: m for m in found_models})
        for n in self:
            ndata = self.graph.node[n]
            if "model_id" in ndata:
                ndata["model"] = models_by_id[ndata["model_id"]]

    def copy(self):
        return self.__copy__()

    def roots(self):
        roots = []
        for n in self.graph:
            if not len(list(self.graph.predecessors(n))):
                roots.append(n)
        return roots

    def leaves(self):
        leaves = []
        for n in self.graph:
            if not len(list(self.graph.successors(n))):
                leaves.append(n)
        return leaves

    def __len__(self):
        return len(self.graph)

    def __iter__(self):
        return self.graph.__iter__()

    def __copy__(self):
        copied = self.__class__(self.browser)
        copied.graph = self.copy_graph(self.graph)
        copied.model_hashes = self.model_hashes
        return copied
