import os

import pytest
import vcr

from pydent import AqSession
from terrarium import AutoPlannerModel

###########
# CONFIG
###########

DEFAULT_NUM_PLANS = 300

###########
# VCR setup
###########

# RECORD_MODE = "new_episodes"
RECORD_MODE = "all"

# TODO: tests: completly deterministic tests
# TODO: tests: parameter or config file for recording mode
# TODO: tests: ignore header in vcr recordings


def hash_response(r):
    """Hash function for request matcher. Defines what vcr will consider
    to be the same request."""
    return "{}:{}:{}".format(r.method, r.uri, r.body)


def hash_test_function(func):
    """Hashes a pytest test function to a unique file name based on
    its class, module, and name"""
    if func.cls:
        cls = func.cls.__name__
    else:
        cls = "None"
    return "{module}_{cls}_{name}".format(
        module=func.module.__name__, cls=cls, name=func.name
    )


def matcher(r1, r2):
    """Request matcher. Defines what vcr considers the same request"""
    return hash_response(r1) == hash_response(r2)


myvcr = vcr.VCR()
myvcr.register_matcher("matcher", matcher)
myvcr.match_on = ["matcher"]
myvcr.record_mode = RECORD_MODE
here = os.path.abspath(os.path.dirname(__file__))
fixtures_path = os.path.join(here, "fixtures/vcr_cassettes")


############
# Test hooks
############


@pytest.hookimpl(hookwrapper=True)
def pytest_pyfunc_call(pyfuncitem):
    """Sorts through each test, uses a vcr cassette to run the test, storing the
    request results into a single file location"""
    cassette_name = hash_test_function(pyfuncitem)
    recordmode = pyfuncitem.keywords._markers.get("record", None)
    if recordmode and recordmode.args[0] != "no":
        myvcr.record_mode = recordmode.args[0]
        with myvcr.use_cassette(os.path.join(fixtures_path, cassette_name) + ".yaml"):
            outcome = yield
    else:
        outcome = yield


###########
# Fixtures
###########


@pytest.fixture(scope="session")
def session(config):
    """
    Returns a live aquarium connection.
    """
    return AqSession(**config)


@pytest.fixture(scope="session")
def datadir():
    here = os.path.dirname(os.path.abspath(__file__))
    dirpath = os.path.join(here, "fixtures", "data")
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    assert os.path.isdir(dirpath), "Data folder does not exist: {}".format(dirpath)
    return dirpath


def new_model(session):
    """Force creation of a new model"""

    with session.with_cache(timeout=60) as sess:
        operation_type = sess.OperationType.find_by_name("Assemble Plasmid")
        sess.Operation.last(500, query=dict(operation_type_id=operation_type.id))
        sess.browser.get("Operation", "plans")
        plans = sess.browser.get("Plan")
    apm = AutoPlannerModel(session.browser, plans)
    apm.set_verbose(True)
    apm.build()
    return apm


@pytest.fixture(scope="function")
def autoplan(session, datadir):
    """The default autoplanner object used in tests. Preferrably loads a pickled
    object. If the pickled object does not exist, a new autoplanner object is created
    and pickled. This object is then unpickled and used."""

    filepath = os.path.join(datadir, "terrarium.pkl")

    if not os.path.isfile(filepath):
        print("TESTS: No file found with path '{}'".format(filepath))
        print("TESTS: Creating new pickled autoplanner...")
        model = new_model(session)
        model.set_verbose(True)
        model.build()
        print("TESTS: dumping {}".format(filepath))
        model.dump(filepath)

    print("TESTS: Loading '{}'".format(filepath))
    model = AutoPlannerModel.load(filepath)
    model.set_verbose(False)
    return model
