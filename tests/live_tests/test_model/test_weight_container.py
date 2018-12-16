import pytest

from autoplanner.model import EdgeWeightContainer, AutoPlannerModel
from pydent.browser import Browser
import dill


@pytest.fixture(scope='module')
def wc(session):

    edge_hash = AutoPlannerModel._hash_afts
    node_hash = AutoPlannerModel._external_aft_hash
    browser = Browser(session)
    plans = browser.last(10, model_class='Plan')
    wc = EdgeWeightContainer(browser, edge_hash, node_hash, plans=plans)
    wc.compute()
    return wc


def test_pickle(wc):
    dill.dumps(wc)


@pytest.fixture(scope="module")
def loaded_wc(wc):
    return dill.loads(dill.dumps(wc))


def test_edge_counter_has_values(loaded_wc):
    assert loaded_wc._edge_counter.counter


def test_node_counter_has_values(loaded_wc):
    assert loaded_wc._node_counter.counter


def test_edge_counter_loaded(loaded_wc, wc):
    assert loaded_wc._edge_counter.counter == wc._edge_counter.counter


def test_node_counter_loaded(loaded_wc, wc):
    assert loaded_wc._node_counter.counter == wc._node_counter.counter


def test_node_and_edge_counter_different(wc, loaded_wc):
    assert wc._node_counter.counter != wc._edge_counter.counter
    assert loaded_wc._node_counter.counter != loaded_wc._edge_counter.counter


def test_node_hash_function_loaded(wc, loaded_wc, session):

    afts = session.AllowableFieldType.last(10)

    counter_attr = '_node_counter'
    counter = getattr(wc, counter_attr).counter
    loaded_counter = getattr(loaded_wc, counter_attr).counter

    prev_counter_data = dict(counter)
    prev_loaded_counter_data = dict(loaded_counter)

    assert prev_counter_data == prev_loaded_counter_data

    for aft in afts:
        counter[aft] += 1
        loaded_counter[aft] += 1

    counter_data = dict(counter)
    loaded_counter_data = dict(loaded_counter)

    assert counter_data != prev_counter_data
    assert loaded_counter_data != prev_loaded_counter_data

    assert counter_data == loaded_counter_data


def test_edge_hash_function_loaded(wc, loaded_wc, session):
    afts = session.AllowableFieldType.last(10)

    counter_attr = '_edge_counter'
    counter = getattr(wc, counter_attr).counter
    loaded_counter = getattr(loaded_wc, counter_attr).counter

    prev_counter_data = dict(counter)
    prev_loaded_counter_data = dict(loaded_counter)

    assert prev_counter_data == prev_loaded_counter_data

    aft_pairs = zip(afts, afts[::-1])

    for pair in aft_pairs:
        counter[pair] += 1
        loaded_counter[pair] += 1

    counter_data = dict(counter)
    loaded_counter_data = dict(loaded_counter)

    assert counter_data != prev_counter_data
    assert loaded_counter_data != prev_loaded_counter_data

    assert counter_data == loaded_counter_data