from terrarium.browser_graph import BrowserGraph
from pydent.browser import Browser


def test_subgraph(session):

    browser = Browser(session)
    graph = BrowserGraph(browser)

    s1, s2 = browser.last(2)

    graph.add_model(s1)
    graph.add_model(s2)

    assert len(graph) == 2

    graph2 = graph.filter_out_models(key=lambda x: x.id != s1.id)

    assert len(graph2) == 1
