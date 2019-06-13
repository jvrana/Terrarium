import dill
from autoplanner import AutoPlannerModel
from pydent.browser import Browser
import os


def test_browser_dumps(session):
    browser = Browser(session)
    browser.last(30)
    dill.dumps(browser)


def test_browser_loads(session):

    browser = Browser(session)
    browser.last(30)
    s = dill.dumps(browser)
    loaded_browser = dill.loads(s)

    assert len(loaded_browser.model_cache) > 0
