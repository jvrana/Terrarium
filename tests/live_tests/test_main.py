import pytest

from autoplanner import AutoPlanner

import pickle
import os

here = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(here, "autoplanner.pkl")


@pytest.fixture(scope="module")
def autoplanner(session):
    ap = AutoPlanner(session)
    ap.construct_template_graph()
    return ap


def test_construct_graph(autoplanner):
    pass


def test_search(autoplanner, session):
    autoplanner.search_graph(session.Sample.one(),
                    session.ObjectType.find_by_name("Yeast Glycerol Stock"),
                    session.ObjectType.find_by_name("Fragment Stock"))