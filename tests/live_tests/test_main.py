import networkx as nx
import os

def cost_function(n, t):
    p = 10e-6
    if t > 0:
        p = n / t
    w = (1 - p) / (1 + p)
    return 10 / (1.000000001 - w)


def test_basic_search(autoplanner, session):
    autoplanner.set_verbose(True)

    ignore_ots = session.OperationType.where({"category": "Control Blocks", "deployed": True})
    ignore = [ot.id for ot in ignore_ots]

    autoplanner.add_model_filter("AllowableFieldType", lambda m: m.field_type.parent_id in ignore)

    autoplanner.search_graph(session.Sample.one(),
                                 session.ObjectType.find_by_name("Yeast Glycerol Stock"),
                                 session.ObjectType.find_by_name("Fragment Stock")
                             )


def test_graphml(autoplanner, datadir):
    nx.write_graphml(autoplanner.template_graph.graph, os.path.join(datadir, 'autoplanner.graphml'))


def test_successor(autoplanner, session):
    graph = autoplanner.template_graph

    aft = autoplanner.browser.find(273, 'AllowableFieldType')
    print(aft)
    print(aft.field_type.name)
    print(aft.field_type.role)
    print(aft.object_type.name)
    print(aft.field_type.operation_type.name)
    node_id = graph.node_id(aft)
    for s in graph.graph.successors(node_id):
        print(s)
        e = graph.get_edge(graph.node_id(aft), s)
        print(e)
        print()

def test_search_new(new_autoplanner, session):
    autoplanner = new_autoplanner
    autoplanner.set_verbose(True)
    autoplanner.search_graph(session.Sample.one(),
                                 session.ObjectType.find_by_name("Yeast Glycerol Stock"),
                                 session.ObjectType.find_by_name("Fragment Stock")
                             )

def test_subgraph(autoplanner, session):

    ot_ids = session.OperationType.find_by_name("Control Blocks")

    for n, ndata in autoplanner.template_graph.iter_model_data("AllowableFieldType"):
        print(ndata['model'].field_type.parent_id)
