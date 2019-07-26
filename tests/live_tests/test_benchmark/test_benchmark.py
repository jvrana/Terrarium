import pytest
from terrarium import AutoPlannerModel
from pydent import Browser


@pytest.mark.record_mode("no")
def test_num_requests(session):

    with session.with_cache() as sess:
        browser = sess.browser

        plans = sess.Plan.last(50)

        n1 = sess._aqhttp.num_requests
        model = AutoPlannerModel(browser, plans)
        assert n1 == sess._aqhttp.num_requests

        sess.set_verbose(True)
        model.build()
        print(sess._aqhttp.num_requests - n1)


@pytest.mark.benchmark
class TestModelBuildBenchmark(object):
    @pytest.mark.record_mode("no")
    def test_build_benchmark(self, benchmark, session):
        session.using_cache = True
        browser = Browser(session)
        session.set_timeout(60)
        plans = session.Plan.last(50)

        def func():
            browser.clear()
            model = AutoPlannerModel(browser, plans)
            model.build()

        benchmark(func)
