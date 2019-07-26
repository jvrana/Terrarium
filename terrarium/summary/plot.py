import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns


def plot_plan_composition(model):
    ops = []

    for p in model.weight_container.plans:
        for op in p.operations:
            ops.append((p, op))

    rows = []
    for p, op in ops:
        rows.append(
            {
                "OperationType": op.operation_type.name,
                "Category": op.operation_type.category,
                "PlanID": p.id,
                "Status": op.status,
            }
        )

    df = pd.DataFrame(rows)
    type_names, type_counts = np.unique(df["OperationType"], return_counts=True)
    argsort = np.argsort(type_counts)
    type_names = type_names[argsort]
    type_counts = type_counts[argsort]

    cat_names, cat_counts = np.unique(df["Category"], return_counts=True)
    argsort = np.argsort(cat_counts)
    cat_names = cat_names[argsort]
    cat_counts = cat_counts[argsort]

    f, axes = plt.subplots(1, 2, figsize=(20, 7), sharex=False)

    ax = axes[0]
    axes[0].set_ylabel("Counts by Type")
    sns.barplot(x=type_names, y=type_counts, ax=ax)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    ax = axes[1]
    ax.set_ylabel("Counts by Category")
    sns.barplot(x=cat_names, y=cat_counts, ax=ax)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    sns.set(style="white")
    sns.despine(left=True)


def edges_df(model, edges):
    edge_counter = model.weight_container._edge_counter
    node_counter = model.weight_container._node_counter

    rows = []
    for n1, n2 in edges:
        if n1 and n2:
            rows.append(
                {
                    "source": "{}_{}".format(n1.id, n1.field_type.operation_type.name),
                    "destination": "{}_{}".format(
                        n2.id, n2.field_type.operation_type.name
                    ),
                    "source_obj": "{}".format(n1.object_type.name),
                    "dest_obj": "{}".format(n1.object_type.name),
                    "count": edge_counter[(n1, n2)],
                    "total": node_counter[n1],
                    "cost": int(model.weight_container.cost(n1, n2)),
                }
            )
    df = pd.DataFrame(rows)
    df.drop_duplicates(inplace=True)
    df["probability"] = df["count"] / df["total"]
    df.sort_values(by=["probability"], inplace=True, ascending=False)
    return df


def explain_operation_type(model, operation_type):
    graph = model.template_graph

    edges = []
    for ft in operation_type.field_types:
        for aft in ft.allowable_field_types:
            node_id = graph.node_id(aft)
            n = graph.get_node(node_id)["model"]
            for successor in graph.successors(node_id):
                s = graph.get_node(successor)["model"]
                edges.append((n, s))
            for pred in graph.predecessors(node_id):
                p = graph.get_node(pred)["model"]
                edges.append((p, n))

    df = edges_df(model, edges)
    dfpivot = df.pivot("source", "destination", "probability")
    f, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=False)
    sns.heatmap(dfpivot, annot=False, cmap="YlGnBu", ax=axes)
