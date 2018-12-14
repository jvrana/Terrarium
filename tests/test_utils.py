from autoplanner.utils import hash_utils
import dill


def test_dump_hash_counter():
    """Expect to be able to dump and load counter"""
    hc = hash_utils.HashCounter(lambda x: str(x**2), [1,2,3,4])

    hc[5] += 1
    hc[6] += 4

    dumped = dill.dumps(hc)
    new_hc = dill.loads(dumped)

    assert dict(hc.counter) == dict(new_hc.counter)


def test_hash_counter():
    """Test hashing for hash counter"""
    data = list(range(7))
    hc1 = hash_utils.HashCounter(lambda x: str(x % 3), data)

    data_dict = {}
    for d in data:
        x = str(d%3)

        if x not in data_dict:
            data_dict[x] = 0
        data_dict[x] += 1

    for k in data:
        assert hc1[k] == data_dict[str(k%3)]


def test_hash_counter_add():
    """Counters should add"""
    hc1 = hash_utils.HashCounter(lambda x: str(x % 2), [1, 2, 3, 4])
    hc2 = hash_utils.HashCounter(lambda x: str(x % 2), [1, 2, 3, 4])
    hc3 = hc1 + hc2
    assert hc3[5] == 4


def test_hash_counter_sub():
    """Counters should subtract"""
    hc1 = hash_utils.HashCounter(lambda x: str(x % 2), [1, 2, 3, 4, 5])
    hc2 = hash_utils.HashCounter(lambda x: str(x % 2), [1, 2, 3, 4])
    hc3 = hc1 - hc2

    assert hc3[5] == 1


def test_hash_counter_mul():
    """Counters should multipler"""
    hc1 = hash_utils.HashCounter(lambda x: str(x % 2), [1, 2, 3, 4, 5])

    hc2 = hc1 * 2
    assert hc2[5] == 6