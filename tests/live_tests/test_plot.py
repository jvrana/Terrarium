from terrarium import summary


def test_plot(autoplan_model):
    summary.plot_plan_composition(autoplan_model)
