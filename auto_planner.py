from pydent.planner import Planner
from pydent.browser import Browser
from mysession import nursery, production, benchapi
from tqdm import tqdm
from uuid import uuid4

from functools import reduce

import networkx as nx

from pydent.utils import filter_list

# Cache relationships
browser = Browser(production)

# determine aft edges
def external_aft_hash(aft):
    if not aft.field_type:
        return str(uuid4())
    if aft.field_type.part:
        part = True
    else:
        part = False
    return "{object_type}-{sample_type}-{part}".format(
        object_type=aft.object_type_id,
        sample_type=aft.sample_type_id,
        part=part,
    )


def internal_aft_hash(aft):
    return "{operation_type}".format(
        operation_type=aft.field_type.parent_id,
        routing=aft.field_type.routing,
        sample_type=aft.sample_type_id
    )

def hash_afts(aft1, aft2):
    source_hash = external_aft_hash(aft1)
    dest_hash = external_aft_hash(aft2)
    return "{}->{}".format(source_hash, dest_hash)
    return str(uuid4())


def cache_afts():
    ots = browser.where({"deployed": True}, "OperationType")
    browser.set_verbose(False)
    production.set_verbose(False)
    results = browser.recursive_retrieve(ots, {
        "field_types": {
            "allowable_field_types": {
                "object_type": [],
                "sample_type": [],
                "field_type": []
            }
        }
    }
    )

    fts = [ft for ft in results['field_types'] if ft.ftype == 'sample']
    inputs = [ft for ft in fts if ft.role == 'input']
    outputs = [ft for ft in fts if ft.role == 'output']

    input_afts = []
    for i in inputs:
        for aft in i.allowable_field_types:
            if aft not in input_afts:
                input_afts.append(aft)

    output_afts = []
    for o in outputs:
        for aft in o.allowable_field_types:
            if aft not in output_afts:
                output_afts.append(aft)

    return input_afts, output_afts


def collect_aft_edges():

    input_afts, output_afts = cache_afts()

    external_groups = {}
    for aft in input_afts:
        external_groups.setdefault(external_aft_hash(aft), []).append(aft)

    internal_groups = {}
    for aft in output_afts:
        internal_groups.setdefault(internal_aft_hash(aft), []).append(aft)

    # from tqdm import tqdm
    edges = []
    for oaft in tqdm(output_afts):
        hsh = external_aft_hash(oaft)
        externals = external_groups.get(hsh, [])
        for aft in externals:
            edges.append((oaft, aft))

    for iaft in tqdm(input_afts):
        hsh = internal_aft_hash(iaft)
        internals = internal_groups.get(hsh, [])
        for aft in internals:
            edges.append((iaft, aft))
    return edges


def cache_edge_weights(plan_depth=100):
    # determine edge weights from plan histories

    plans = browser.last(plan_depth, "Plan")
    browser.recursive_retrieve(plans, {
        "operations": {"field_values": ["allowable_field_type", "wires_as_source", "wires_as_dest"]}}, strict=False)
    for p in plans:
        p.wires

    all_wires = reduce((lambda x, y: x + y), [p.wires for p in plans])
    all_operations = reduce((lambda x, y: x + y), [p.operations for p in plans])
    browser.recursive_retrieve(all_wires[:], {
        "source": {
            "field_type": [],
            "operation": "operation_type",
            "allowable_field_type": []
        },
        "destination": {
            "field_type": [],
            "operation": "operation_type",
            "allowable_field_type": []
        }
    })
    return {
        "wires": all_wires,
        "operations": all_operations
    }

def edge_cost_function(num, total):



def determine_historical_edge_weights(history_depth=100):
    results = cache_edge_weights(history_depth)

    wire_hash_count = {}
    wire_source_hash_count = {}

    aft_pairs = []
    for wire in results['wires']:
        if wire.source and wire.destination:
            aft_pairs.append((wire.source.allowable_field_type, wire.destination.allowable_field_type))
    for op in results['operations']:
        for i in op.inputs:
            for o in op.outputs:
                aft_pairs.append((i.allowable_field_type, o.allowable_field_type))

    for aft1, aft2 in tqdm(aft_pairs):
        if aft1 and aft2:
            aft_hash = hash_afts(aft1, aft2)
            wire_hash_count.setdefault(aft_hash, 0)
            wire_hash_count[aft_hash] += 1

            source_hash = external_aft_hash(aft1)
            wire_source_hash_count.setdefault(source_hash, 0)
            wire_source_hash_count[source_hash] += 1

    def weight_edge(aft1, aft2):
        p = 1e-4
        if aft1 and aft2:
            #         if aft1.field_type.role == 'input' and aft2.field_type.role == 'output':
            #             return 0
            n = wire_hash_count.get(hash_afts(aft1, aft2), 0) * 1.0
            t = wire_source_hash_count.get(external_aft_hash(aft1), 0)
            if t > 0:
                p = n/t
            return 10 / (1.001 - ((1.0 - n / t) / (1.0 + n / t)))
        return default

    return weight_edge


def generate_graph():

    # new direction graph
    G = nx.DiGraph()
    edges = collect_aft_edges()

    all_afts = []
    for aft1, aft2 in all_afts:
        if aft1 not in all_afts:
            all_afts.append(aft1)
        if aft2 not in all_afts:
            all_afts.append(aft2)

    weight_func = determine_historical_edge_weights(100)

    # add weighted edges
    for aft1, aft2 in edges:
        if aft1 and aft2:
            G.add_edge(aft1.id, aft2.id, weight=weight_func(aft1, aft2))

    print("Building Graph:")
    print("  {} edges".format(len(edges)))
    print("  {} nodes".format(len(G)))


    ignore_ots = browser.where({"category": "Control Blocks"}, "OperationType")
    # ignore_ots = []
    nodes = [aft.id for aft in all_afts if
             aft.field_type.parent_id and aft.field_type.parent_id not in [ot.id for ot in ignore_ots]]
    graph = G.subgraph(nodes)

    print("Graph size reduced from {} to {} nodes".format(len(G), len(graph)))

    print("Example edges")
    for e in list(graph.edges)[:3]:
        aft1 = browser.find(e[0], "AllowableFieldType")
        aft2 = browser.find(e[1], "AllowableFieldType")
        print()
        print("{} {}".format(aft1.field_type.role, aft1))
        print("{} {}".format(aft2.field_type.role, aft2))

    for edge in list(graph.edges)[:5]:
        print(graph[edge[0]][edge[1]])
    return graph


def search_graph(graph, obj1, obj2, input_afts, output_afts):

    # obj1 = production.ObjectType.find_by_name("Plasmid Stock")
    # obj2 = production.ObjectType.find_by_name("Yeast Glycerol Stock")

    afts1 = filter_list(input_afts, object_type_id=obj1.id, sample_type_id=obj1.sample_type_id)
    afts2 = filter_list(output_afts, object_type_id=obj2.id, sample_type_id=obj2.sample_type_id)

    start_nodes = [aft.id for aft in afts1 if aft.id in graph]
    end_nodes = [aft.id for aft in afts2 if aft.id in graph]

    print("Input afts:")
    for aftid in start_nodes:
        aft = browser.find(aftid, 'AllowableFieldType')
        try:
            print(aft.field_type.operation_type)
        except:
            pass

    allpaths = []
    for n1 in tqdm(start_nodes):
        for n2 in end_nodes:
            try:
                path = nx.dijkstra_path(graph, n1, n2, weight='weight')
                path_length = nx.dijkstra_path_length(graph, n1, n2, weight='weight')
                allpaths.append((path, path_length))
            except nx.exception.NetworkXNoPath:
                pass

    allpaths = sorted(allpaths, key=lambda x: x[1])

    print()
    print("*" * 50)
    print("{} >> {}".format(obj1.name, obj2.name))
    print("*" * 50)
    print()
    for path, pathlen in allpaths[:10]:
        print(path)
        ots = []
        for aftid in path:
            aft = browser.find(aftid, 'AllowableFieldType')
            ot = browser.find(aft.field_type.parent_id, 'OperationType')
            ots.append("{ot}".format(role=aft.field_type.role, name=aft.field_type.name, ot=ot.name))

        edge_weights = [graph[x][y]['weight'] for x, y in zip(path[:-1], path[1:])]
        print(pathlen)
        print(edge_weights)
        print(len(ots))
        print(ots)
        print()
