# @markdown TODO: remove trident dependencies, pull only relevant data

from more_itertools import spy


def sample_type_subgraph(g, stid):
    node_ids = []
    for nid, ndata in g.nodes(data=True):
        if ndata["sample_type_id"] == stid:
            node_ids.append(nid)
    return g.subgraph(node_ids)


def samples_to_parent_graph(sample, g=None, visited=[]):
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


def connect_sample_graphs(g1, g2):
    _, output_afts = collect_input_output_afts(g1.iter_models("AllowableFieldType"))
    input_afts, _ = collect_input_output_afts(g2.iter_models("AllowableFieldType"))
    edges = match_afts(input_afts, output_afts, internal_aft_hash)
    return edges


def build_graph(template_graph, sample_graph):
    sample_graphs = {}
    for sample in sample_graph.iter_models(model_class="Sample"):
        g = sample_type_subgraph(template_graph, sample.sample_type_id)
        g.set_prefix("Sample{}_".format(sample.id))
        sample_graphs[sample.id] = g

    graph = BrowserGraph(template_graph.browser)
    graph._graph = nx.compose_all([sg.graph for sg in sample_graphs.values()])

    for x in sample_graph.edges():
        s1 = sample_graph.get_model(x[0])
        s2 = sample_graph.get_model(x[1])

        g1 = sample_graphs[s1.id]
        g2 = sample_graphs[s2.id]
        edges = connect_sample_graphs(g1, g2)
        for e in edges:
            n1 = template_graph.node_id(e[0])
            n2 = template_graph.node_id(e[1])
            edge = template_graph.get_edge(
                template_graph.node_id(e[0]), template_graph.node_id(e[1])
            )
            graph.graph.nodes[n1]["sample"] = s1
            graph.graph.nodes[n2]["sample"] = s2
            graph.graph.add_edge(
                n1, n2, weight=edge["weight"], edge_type="sample_to_sample"
            )
    return graph


s = session.Sample.find(27608)
graph = build_graph(template_graph, samples_to_parent_graph(s))
