from terrarium.schemas.validate import validate_with_schema


def extract_end_nodes(graph, goal_sample_id, goal_object_type_id):
    """
    Extract any afts that match the goal_sample_id and goal_object_type_id
    :param graph:
    :param goal_sample_id:
    :param goal_object_type_id:
    :return:
    """

    match = {
        "sample": {"id": goal_sample_id},
        "object_type_id": goal_object_type_id,
        "field_type": {"role": "output"},
    }

    end_nodes = []
    for n, aft in graph.model_data("AllowableFieldType"):
        if validate_with_schema(aft, match):
            end_nodes.append(n)
    return end_nodes


def extract_items(graph):
    items = [ndata for _, ndata in graph.model_data("Item")]
    return items


def clean_graph(self, graph):
    """
        Remove internal wires with different routing id but same sample

        :param graph:
        :type graph:
        :return:
        :rtype:
        """
    removal = []

    afts = graph.models("AllowableFieldType")
    removal = []
    for n1, n2 in graph.edges:
        d1 = graph.get_data(n1)
        d2 = graph.get_data(n2)

        input_schema = {
            "__class__": "AllowableFieldType",
            "field_type": {"role": "input", "routing": str},
            "sample": dict,
        }

        output_schema = {
            "__class__": "AllowableFieldType",
            "field_type": {"role": "output", "routing": str},
            "sample": dict,
        }

        if validate_with_schema(d1, input_schema) and validate_with_schema(
            d2, output_schema
        ):
            schema = {
                "sample": {"id": d2["sample"]["id"]},
                "field_type": {"routing": d2["field_type"]["routing"]},
            }
            if not validate_with_schema(d1, schema):
                removal.append((n1, n2))

    graph.graph.remove_edges_from(removal)
    return graph
