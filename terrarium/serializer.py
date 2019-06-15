class Serializer(object):

    @staticmethod
    def serialize(model, *args, **kwargs):
        data = model.dump(*args, **kwargs)
        data["primary_key"] = model._primary_key
        data["__class__"] = model.__class__.__name__
        return data

    @classmethod
    def serialize_aft(cls, aft):
        data = cls.serialize(aft, include="field_type")
        return data

    @classmethod
    def serialize_all_afts(cls, session):
        session.OperationType.where({"deployed": True})

        session.browser.get(
            "OperationType",
            {
                "field_types": {
                    "allowable_field_types": {
                        "object_type",
                        "sample_type",
                        "field_type",
                    }
                }
            },
        )
        afts = session.browser.get("AllowableFieldType")
        return [cls.serialize_aft(aft) for aft in afts]

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
