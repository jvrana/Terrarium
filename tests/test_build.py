from terrarium import SampleGraphBuilder, ProtocolBlueprintBuilder, ProtocolGraphBuilder

def test_build(session):
    s = session.Sample.find(27608)
    sample_graph = SampleGraphBuilder.build([s])

    blueprint = ProtocolBlueprintBuilder()
    blueprint.build_template_graph()