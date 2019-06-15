import pytest
from pydent import AqSession


@pytest.fixture(scope="session")
def base_session():
    return AqSession("vrana", "Mountain5", "http://0.0.0.0")


@pytest.fixture
def session(base_session):
    yield base_session()
