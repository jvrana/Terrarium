from copy import deepcopy
from terrarium.builders.utils import match_afts
from terrarium.builders.hashes import external_aft_hash, internal_aft_hash
from terrarium import constants as C

aft = {
    C.MODEL_CLASS: "AllowableFieldType",
    "field_type": {"part": False, "array": False, "parent_id": 1},
    "field_type_id": 2,
    "sample_type_id": 2,
    "object_type_id": 3,
}


class TestExternalHash:
    def test_match(self):
        aft1 = deepcopy(aft)
        assert list(match_afts([aft1], [aft], external_aft_hash))

    def test_part_mismatch(self):
        aft1 = deepcopy(aft)
        aft1["field_type"]["part"] = True
        assert not list(match_afts([aft1], [aft], external_aft_hash))

    def test_array_match(self):
        aft1 = deepcopy(aft)
        aft1["field_type"]["array"] = True
        assert list(match_afts([aft1], [aft], external_aft_hash))

    def test_otid_mismatch(self):
        aft1 = deepcopy(aft)
        aft1["object_type_id"] = 999
        assert not list(match_afts([aft1], [aft], external_aft_hash))

    def test_stid_mismatch(self):
        aft1 = deepcopy(aft)
        aft1["sample_type_id"] = 999
        assert not list(match_afts([aft1], [aft], external_aft_hash))


class TestInternalHash:
    def test_match(self):
        aft1 = deepcopy(aft)
        assert list(match_afts([aft1], [aft], internal_aft_hash))

    def test_parent_id_mismatch(self):
        aft1 = deepcopy(aft)
        aft1["field_type"]["parent_id"] = 999
        assert not list(match_afts([aft1], [aft], internal_aft_hash))
