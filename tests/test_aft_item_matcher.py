from terrarium.utils.validate import validate_with_schema_errors, is_in, is_any_type_of
from copy import deepcopy
import pytest
from terrarium.utils.matcher import aft_matches_item


class TestMatchAFT(object):
    @pytest.fixture
    def aft_data(self, session):
        aft = session.AllowableFieldType.one()
        aft_data = aft.dump(include={"field_type"})
        aft_data["field_type"]["part"] = aft_data["field_type"]["part"] == True
        return aft_data

    @pytest.fixture
    def item_data(self, session):
        return session.Item.one().dump(include="sample")

    @pytest.fixture
    def true_item(self, item_data, aft_data):
        true_item = deepcopy(item_data)
        true_item["object_type_id"] = aft_data["object_type_id"]
        true_item["sample"]["sample_type_id"] = aft_data["sample_type_id"]
        true_item["is_part"] = aft_data["field_type"]["part"]
        return true_item

    def test_does_match(self, true_item, aft_data):
        assert aft_matches_item(aft_data, true_item)

    def test_wrong_object_type_id(self, true_item, aft_data):
        false_item = deepcopy(true_item)
        false_item["object_type_id"] += 1
        assert not aft_matches_item(aft_data, false_item)

    def test_wrong_sample_type_id(self, true_item, aft_data):
        false_item = deepcopy(true_item)
        false_item["sample"]["sample_type_id"] += 1
        assert not aft_matches_item(aft_data, false_item)

    def test_aft_stid_is_none(self, true_item, aft_data):
        aft_data_copy = deepcopy(aft_data)
        aft_data_copy["sample_type_id"] = None
        false_item = deepcopy(true_item)
        false_item["sample"]["sample_type_id"] += 1

        assert not aft_matches_item(aft_data, false_item)
        assert aft_matches_item(aft_data_copy, false_item)

    def test_aft_otid_is_none(self, true_item, aft_data):
        aft_data_copy = deepcopy(aft_data)
        aft_data_copy["object_type_id"] = None
        false_item = deepcopy(true_item)
        false_item["object_type_id"] += 1

        assert not aft_matches_item(aft_data, false_item)
        assert aft_matches_item(aft_data_copy, false_item)
