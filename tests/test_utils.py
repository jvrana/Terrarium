from autoplanner.utils import hash_utils
import dill


def test_dump_hash_counter():
    hc = hash_utils.HashCounter(lambda x: str(x**2), [1,2,3,4])

    hc[5] += 1
    hc[6] += 1

    dumped = dill.dumps(hc)
    dill.loads(dumped)