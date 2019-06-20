from terrarium.graphs.graphs import SchemaGraph, SchemaValidationError
import pytest
import json


def test_schema_graph_init():
    f = lambda d: json.dumps(d)
    g = SchemaGraph(f)
    assert g is not None
    assert g.graph is not None


def test_schema_graph():
    f = lambda d: json.dumps(d)
    g = SchemaGraph(f)
    g.schemas.append({"name": str, "id": int})
    g.add_data({"name": "anyong", "id": 4})


def test_schema_graph_raises():
    f = lambda d: json.dumps(d)
    g = SchemaGraph(f)
    with pytest.raises(SchemaValidationError):
        g.schemas.append({"name": str, "id": str})
        g.add_data({"name": "anyong", "id": 4})


def test_schema_id():
    g = SchemaGraph(lambda d: d["name"])
    g.schemas = []
    g.add_data({"name": "anyong"})
    assert "anyong" in g
