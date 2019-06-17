from terrarium.utils.validate import validate_with_schema_errors, is_in


def aft_matches_item(aft_data, item_data):
    return validate_with_schema_errors(
        aft_data,
        {
            "field_type": {"part": item_data["is_part"]},
            "sample_type_id": is_in([None, item_data["sample"]["sample_type_id"]]),
            "object_type_id": is_in([None, item_data["object_type_id"]]),
        },
    )[0]
