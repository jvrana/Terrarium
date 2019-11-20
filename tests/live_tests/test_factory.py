from terrarium import NetworkFactory


def test_plan_existing_item(autoplan_model, session):
    factory = NetworkFactory(autoplan_model)

    yeast_glycerol_stock = session.ObjectType.find_by_name("Yeast Glycerol Stock")

    item = session.Item.one(query={"object_type_id": yeast_glycerol_stock.id})

    network = factory.new_from_sample(item.sample)

    solution = network.run(goal_object_type=item.object_type)

    print(solution)
