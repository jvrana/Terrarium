# TODO: this should be the only class that makes requests
# TODO: this should take in a session object and reuse that object throughout
from terrarium.schemas.schemas import Schema

from functools import wraps
from terrarium.schemas.validate import validate_with_schema_errors
from terrarium.exceptions import SchemaValidationError


def validate(schema):
    def validate_function(f):
        @wraps(f)
        def validation_wrapper(*args, **kwargs):
            result = f(*args, **kwargs)
            valid, errors = validate_with_schema_errors(result, schema)
            if errors:
                SchemaValidationError.raise_from_errors("Data was invalid", errors)
            return result

        return validation_wrapper

    return validate_function


class Serializer(object):
    """
    Serializes models.
    """

    @staticmethod
    def serialize(model, *args, **kwargs):
        data = model.dump(*args, **kwargs)
        data[Schema.PRIMARY_KEY] = model._primary_key
        data[Schema.MODEL_CLASS] = model.__class__.__name__
        return data

    @classmethod
    def serialize_aft(cls, aft):
        data = cls.serialize(aft, include="field_type")
        return data

    @classmethod
    def serialize_plans(cls, plans):
        session = plans[0].session
        session.browser.get("Plan", {"operations": {"field_values"}})
        session.browser.get("Wire", {"source", "destination"})
        session.browser.get("FieldValue", {"allowable_field_type": "field_type"})
        session.browser.get("Operation", "operation_type")
        nodes = [
            cls.serialize_aft(aft) for aft in session.browser.get("AllowableFieldType")
        ]
        edges = []
        for w in session.browser.get("Wire"):
            src_aft = cls.serialize_aft(w.source.allowable_field_type)
            dest_aft = cls.serialize_aft(w.destination.allowable_field_type)
            edges.append((src_aft, dest_aft))
        return nodes, edges
