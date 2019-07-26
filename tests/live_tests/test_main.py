import networkx as nx
import os


def test_basic_search(autoplan, session):
    autoplan.set_verbose(True)

    ignore_ots = session.OperationType.where(
        {"category": "Control Blocks", "deployed": True}
    )
    ignore = [ot.id for ot in ignore_ots]

    autoplan.add_model_filter(
        "AllowableFieldType", lambda m: m.field_type.parent_id in ignore
    )

    autoplan.search_graph(
        session.Sample.one(),
        session.ObjectType.find_by_name("Yeast Glycerol Stock"),
        session.ObjectType.find_by_name("Fragment Stock"),
    )


def test_graphml(autoplan, datadir):
    nx.write_graphml(
        autoplan.template_graph.graph, os.path.join(datadir, "autoplan.graphml")
    )


def test_successor(autoplan):
    graph = autoplan.template_graph

    aft = autoplan.browser.find(273, "AllowableFieldType")
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


# def test_search_new(new_autoplanner, session):
#     autoplanner = new_autoplanner
#     autoplan.set_verbose(True)
#     autoplan.search_graph(session.Sample.one(),
#                                  session.ObjectType.find_by_name("Yeast Glycerol Stock"),
#                                  session.ObjectType.find_by_name("Fragment Stock")
#                              )