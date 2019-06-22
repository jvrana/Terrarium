import arrow
from more_itertools import flatten

from terrarium.adapters.aquarium import Serializer
from terrarium.graphs import ModelGraph


def save_events(session, g):
    def timestamp(s):
        return int(s / 60.0 / 60.0)

    now = arrow.now().timestamp

    events = []  # list of events that you can categorize

    for _plan in session.browser.get("Plan"):
        t = timestamp(arrow.get(_plan.created_at).timestamp)
        events.append((t, "Plan was created"))

    session.browser.get("Operation", "jobs")
    for j in session.browser.get("Job"):
        t = timestamp(arrow.get(j.created_at).timestamp)
        events.append((t, "Protocol job created"))

        t = timestamp(arrow.get(j.updated_at).timestamp)
        events.append((t, "Protocol job finished"))

    for n, ndata in g.node_data():
        c = ndata["__class__"]
        if c == "DataAssociation":
            t = timestamp(arrow.get(ndata["created_at"]).timestamp)
            events.append((t, "Data associated to {}".format(ndata["parent_class"])))
            if ndata["key"] == "status_change":
                t = timestamp(arrow.get(ndata["created_at"]).timestamp)
                events.append((t, "User change operation status"))
        elif c == "Sample":
            pass
        elif c == "Item":
            t = timestamp(arrow.get(ndata["created_at"]).timestamp)
            events.append((t, "Item created"))

            if ndata["location"] == "deleted":
                t = timestamp(arrow.get(ndata["updated_at"]).timestamp)
                events.append((t, "Item deleted"))
        elif c == "Operation":
            t = timestamp(arrow.get(ndata["created_at"]).timestamp)
            events.append((t, "Operation created"))
            if ndata["status"] == "done":
                t = timestamp(arrow.get(ndata["updated_at"]).timestamp)
                events.append((t, "Operation completed"))
            elif ndata["status"] == "error":
                t = timestamp(arrow.get(ndata["updated_at"]).timestamp)
                events.append((t, "Operation errored"))

        g.node[n]["start"] = timestamp(arrow.get(ndata["created_at"]).timestamp)
        g.node[n]["end"] = timestamp(now)

    event_data = {}
    for t, e in events:
        event_data.setdefault(e, []).append(t)

    import json

    with open("events.json", "w") as f:
        json.dump(event_data, f)


def create_graph(session):
    serialize = Serializer.serialize

    field_values = []
    for plan in session.browser.get("Plan"):
        for op in plan.operations:
            field_values += op.field_values

    wires = session.browser.get("Wire")
    assert wires
    items = session.browser.get("Item")
    assert items

    g = ModelGraph()

    field_values = []
    for plan in session.browser.get("Plan"):
        for da in plan.data_associations:
            g.add_edge_from_models(serialize(plan), serialize(da))

    #### ADDERS

    add_edge = lambda g, a, b: g.add_edge_from_models(serialize(a), serialize(b))
    add_rev_edge = lambda g, a, b: g.add_edge_from_models(serialize(b), serialize(a))

    def add_fv(graph, fv, f=add_edge):
        if fv.sample:
            f(graph, fv, fv.sample)
            if fv.item:
                f(graph, fv.sample, fv.item)
                return fv.item
            return fv.sample
        elif fv.item:
            f(fv, fv.item)
            return fv.item

    ### END ADDERS

    for item in items:
        for da in item.data_associations:
            add_edge(g, item, da)

    for plan in session.browser.get("Plan"):
        for op in plan.operations:
            add_edge(g, plan, op)
            for fv in op.inputs:
                add_edge(g, fv, op)
            for fv in op.outputs:
                add_edge(g, op, fv)
            for da in op.data_associations:
                add_edge(g, op, da)
            for fv in field_values:
                if fv.role == "input":
                    add_fv(g, fv, f=add_rev_edge)
                elif fv.role == "output":
                    add_fv(g, fv, f=add_edge)
                else:
                    add_edge(op, fv)

    for w in wires:
        m1 = add_fv(g, w.source, f=add_edge)
        m2 = add_fv(g, w.destination, f=add_rev_edge)
        add_edge(g, w.source, w)
        add_edge(g, w, w.destination)
        add_edge(g, m1, w)
        add_edge(g, w, m2)
    return g


def test_timeline(base_session):
    session = base_session.with_cache(timeout=60)

    session.Plan.find([30667, 30668, 30666, 32551, 30380])

    session.browser.get(
        "Plan",
        {
            "operations": {"field_values": [], "operation_type": {"field_types"}},
            "data_associations": [],
        },
    )
    wires = flatten([p.wires for p in session.browser.get("Plan")])
    session.browser.update_cache(list(wires))
    session.browser.get("Wire", ["source", "destination"])
    session.browser.get("Operation", ["data_associations"])
    session.browser.get(
        "FieldValue",
        {"sample": "sample_type", "item": ["object_type", "data_associations"]},
    )

    g = create_graph(session)
    save_events(session, g)

    g.write_gexf("timeline.gexf")
