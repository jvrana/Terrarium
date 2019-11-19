from copy import deepcopy

import pytest
from pydent.browser import Browser

from terrarium import AutoPlannerModel


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
