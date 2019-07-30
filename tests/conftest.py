from pydent import AqSession
from os.path import abspath, dirname, join, isdir
from os import mkdir
import vcr
import pytest
from terrarium.utils import logger

logger.set_level("INFO")

here = dirname(abspath(__file__))


@pytest.fixture
def make_data_dir():
    """Makes a new data directory in the fixtures."""
    fixtures = join(here, "fixtures")
    if not isdir(fixtures):
        mkdir(fixtures)

    def make_directory(name):
        path = join(here, "fixtures", name)
        if not isdir(path):
            mkdir(path)
        return path

    return make_directory


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers", "recordmode(mode): mark test to have its requests recorded"
    )


def pytest_addoption(parser):
    parser.addoption(
        "--webtest", action="store_true", default=False, help="run web tests"
    )
    parser.addoption(
        "--recordmode",
        action="store",
        default="no",
        help="[no, all, new_episodes, once, none]",
    )


def pytest_collection_modifyitems(config, items):
    skip_web = pytest.mark.skip(reason="need --webtest option to run")
    record_mode = pytest.mark.record(config.getoption("--recordmode"))
    for item in items:
        if config.getoption("--recordmode") != "no":
            item.add_marker(record_mode)


###########
# VCR setup
###########


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


############
# Test hooks
############

# https://vcrpy.readthedocs.io/en/latest/usage.html
myvcr = vcr.VCR()
myvcr.register_matcher("matcher", matcher)
myvcr.match_on = ["matcher"]
# record mode is handled in pytest.ini
here = abspath(dirname(__file__))
fixtures_path = join(here, "fixtures/vcr_cassettes")

USE_VCR = True


@pytest.hookimpl(hookwrapper=True)
def pytest_pyfunc_call(pyfuncitem):
    """Sorts through each test, uses a vcr cassette to run the test, storing the
    request results into a single file location"""
    cassette_name = hash_test_function(pyfuncitem)

    markers = pyfuncitem.own_markers

    record_modes = []
    for marker in markers:
        record_modes += marker.args

    if "no" not in record_modes and USE_VCR:
        myvcr.record_mode = record_modes[0]
        with myvcr.use_cassette(join(fixtures_path, cassette_name) + ".yaml"):
            outcome = yield  # runs the test
    else:
        outcome = yield  # runs the test


@pytest.fixture(scope="session")
def base_session():
    return AqSession("vrana", "Mountain5", "http://0.0.0.0")


@pytest.fixture
def session(base_session):
    yield base_session()
