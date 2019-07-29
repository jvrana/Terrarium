import pytest


EXAMPLE_SAMPLE_ID = 27608


@pytest.fixture(scope="module")
def example_sample(base_session):
    return base_session.Sample.find(EXAMPLE_SAMPLE_ID)
