from terrarium.schemas.validate import is_any_type_of
from terrarium import constants as C
from copy import deepcopy


class Schema(object):

    MODEL_CLASS = C.MODEL_CLASS
    PRIMARY_KEY = C.PRIMARY_KEY

    def __init__(self, model_class=str, schema_data=None):
        schema = {self.MODEL_CLASS: model_class, self.PRIMARY_KEY: int}
        if schema_data:
            schema.update(schema_data)
        self.schema = schema

    def update(self, schema: dict):
        self.schema.update(schema)

    def copy(self):
        return deepcopy(self)


class CustomSchemas(object):

    FT_SCHEMA = {
        "role": str,
        "ftype": str,
        "part": is_any_type_of(None, bool),
        "parent_id": int,
        "array": is_any_type_of(None, bool),
        "routing": is_any_type_of(None, str),
    }

    SAMPLE_SCHEMA = Schema("Sample", {"id": int, "sample_type_id": int})
    AFT_SCHEMA = Schema(
        "AllowableFieldType",
        {
            "object_type_id": is_any_type_of(int, None),
            "sample_type_id": is_any_type_of(int, None),
            "field_type_id": int,
            "field_type": dict(
                operation_type=dict(name=str, field_types=[FT_SCHEMA]), **FT_SCHEMA
            ),
        },
    )

    ITEM_SCHEMA = Schema(
        "Item",
        {
            "is_part": bool,
            "object_type_id": is_any_type_of(int, None),
            "collections": [{"object_type_id": int}],
        },
    )
    AFT_SAMPLE_SCHEMA = AFT_SCHEMA.copy()
    AFT_SAMPLE_SCHEMA.update({"sample": {"id": int, "sample_type_id": int}})

    aft_sample_schema = AFT_SAMPLE_SCHEMA.schema
    aft_schema = AFT_SCHEMA.schema
    item_schema = ITEM_SCHEMA.schema
    sample_schema = SAMPLE_SCHEMA.schema
