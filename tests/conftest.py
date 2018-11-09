import json
import os

import pytest


# TODO: use os.environ by default, fall back to config.json.secret
@pytest.fixture(scope="session")
def config():
    """
    Returns the config dictionary for live tests.
    """
    dir = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(dir, "secrets", "config.json.secret")
    with open(config_path, 'rU') as f:
        config = json.load(f)
    return config
