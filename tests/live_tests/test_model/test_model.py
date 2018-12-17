import pytest

from autoplanner.model import ModelFactory


@pytest.fixture(scope='function')
def factory(session):
    return ModelFactory(session)


@pytest.fixture(scope='module')
def users(session):
    NUM_PLANS = 30
    plans = session.Plan.last(NUM_PLANS)
    user_ids = set(p.user_id for p in plans)
    assert len(user_ids) > 1
    users = session.User.find(list(user_ids))
    return users


def test_empty_model(factory):
    model = factory.new()
    assert model


@pytest.mark.parametrize('num', [1, 11])
def test_model_factory_new(factory, num):
    """Expect factory to pull in exactly the specified number of plans"""
    model = factory.new(num)
    assert len(model.weight_container.plans) == num


def test_model_factory_emulate_single_user(factory, config, session):
    """Expect emulate to pull in specified number of plans, all belonging
    to the single user"""
    login = config['login']
    user = session.User.one(login=login)
    model = factory.emulate(login, limit=10)
    assert len(model.weight_container.plans) == 10
    for p in model.weight_container.plans:
        assert p.user_id == user.id


def test_model_factory_emulate_multiple_users(factory, users):
    model = factory.emulate([u.login for u in users], limit=30)
    model_plans = model.weight_container.plans
    user_ids_of_plans = set(p.user_id for p in model_plans)
    assert len(model_plans) == 30
    assert len(user_ids_of_plans) == len(users)


def test_model_attributes(factory):
    """Models should have a 'version', 'name', 'updated'"""

    model = factory.new(1)
    assert hasattr(model, 'version')
    assert hasattr(model, 'name')
    assert hasattr(model, 'created_at')
    assert hasattr(model, 'updated_at')


def test_model_info(factory):
    model = factory.new(1)
    model.print_info()


def test_build(factory):
    """Build a small, new model"""
    model = factory.new(10)
    model.build()


def test_basic_search(autoplanner, session):
    autoplanner.set_verbose(True)

    ignore_ots = session.OperationType.where({"category": "Control Blocks", "deployed": True})
    ignore = [ot.id for ot in ignore_ots]

    autoplanner.add_model_filter("AllowableFieldType", lambda m: m.field_type.parent_id in ignore)

    autoplanner.search_graph(session.Sample.one(),
                             session.ObjectType.find_by_name("Yeast Glycerol Stock"),
                             session.ObjectType.find_by_name("Fragment Stock")
                             )


def test_model_saves_and_load(factory, tmpdir):
    model = factory.new(10)
    model.build()
    path = tmpdir.mkdir("models").join("test_model.pkl")
    model.save(path)

    loaded = factory.load_model(path)
    assert len(loaded.weight_container._edge_counter.counter) > 0
    assert len(loaded.weight_container._node_counter.counter) > 0


def test_model_add(factory, users):
    models = [factory.emulate(u.login, limit=10) for u in users[:3]]
    for m in models:
        m.print_info()
        m.build()
    new_model = factory.new()
    for m in models:
        new_model += m

    for k, v in new_model.weight_container._edge_counter.counter.items():
        summed = 0
        for m in models:
            summed += m.weight_container._edge_counter.counter[k]
        assert v == summed

    for k, v in new_model.weight_container._node_counter.counter.items():
        summed = 0
        for m in models:
            summed += m.weight_container._node_counter.counter[k]
        assert v == summed

    assert new_model


def test_model_mul(factory, users):
    models = [factory.emulate(u.login, limit=10) for u in users[:1]]
    m = models[0]
    m.build()
    new_model = m * 3


def test_model_compose_complex(factory, users):
    models = [factory.emulate(u.login, limit=10) for u in users[:4]]
    for m in models:
        m.print_info()
        m.build()

    m1 = models[0]
    m2 = models[1]
    m3 = models[3]

    m4 = m1 + m2 * 3 + m3


def test_model_sub():
    assert False, "Test not built yet..."
