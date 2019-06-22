from terrarium.graphs import AFTGraph
import networkx as nx


def test_multigraph():
    g = AFTGraph()
    gcopy = g.copy().to(nx.MultiDiGraph)
