import json
import os

import pytest


# TODO: use os.environ by default, fall back to config.json.secret
@pytest.fixture(scope="session")
def config():
    """Returns the config dictionary for live tests."""
    dir = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(dir, "secrets", "config.json.secret")
    with open(config_path, "rU") as f:
        config = json.load(f)
    return config


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
    skip_slow = pytest.mark.skip(reason="need --webtest option to run")
    record_mode = pytest.mark.record(config.getoption("--recordmode"))
    for item in items:
        if config.getoption("--recordmode") != "no":
            item.add_marker(record_mode)
