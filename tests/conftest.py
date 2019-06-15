import pytest
from pydent import AqSession


@pytest.fixture(scope="function")
def session():
    local = AqSession("vrana", "Mountain5", "http://0.0.0.0")
    return local
