import pytest
from terrarium import AutoPlannerModel
from copy import deepcopy
from pydent.browser import Browser


@pytest.fixture(scope="module")
def example_plans(session):
    return session.Plan.last(10)


@pytest.fixture(scope="function")
def plans(example_plans):
    return deepcopy(example_plans)


def test_init_from_plans(session, plans):
    model = AutoPlannerModel(Browser(session), plans=plans)
    assert model


def test_model_bulid(session, plans):
    model = AutoPlannerModel(Browser(session), plans=plans)
    model.build()
