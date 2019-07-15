from typing import Sequence
from terrarium.utils import dict_intersection, group_by
from itertools import product, chain
from more_itertools import flatten
from terrarium.schemas.validate import validate_with_schema_errors, is_in


def match_afts(afts1: Sequence[dict], afts2: Sequence[dict], hash_function: callable):
    group1 = group_by(afts1, hash_function)
    group2 = group_by(afts2, hash_function)

    d = dict_intersection(group1, group2, lambda a, b: product(a, b))
    edges = chain(*flatten((d.values())))
    return edges


def aft_matches_item(aft_data: dict, item_data: dict):
    return validate_with_schema_errors(
        aft_data,
        {
            "field_type": {"part": item_data["is_part"]},
            "sample_type_id": is_in([None, item_data["sample"]["sample_type_id"]]),
            "object_type_id": is_in([None, item_data["object_type_id"]]),
        },
    )[0]
