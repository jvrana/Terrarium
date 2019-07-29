import pytest
from os.path import abspath, dirname, basename

here = abspath(dirname(__file__))


@pytest.fixture
def datadir(make_data_dir):
    return make_data_dir(basename(dirname(__file__)))
