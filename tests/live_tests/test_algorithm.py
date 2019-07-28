import networkx as nx
from pydent.browser import Browser
from terrarium.model import AutoPlannerModel
from terrarium.network import NetworkOptimizer


def test_autoplan(autoplan_model, session):
    ots = session.OperationType.where({"category": "Control Blocks"})
    assert ots
    print(autoplan_model.template_graph.graph.number_of_nodes())
    autoplan_model.add_model_filter(
        "AllowableFieldType",
        "exclude",
        lambda x: x.field_type.parent_id in [ot.id for ot in ots],
    )
    print(autoplan_model.template_graph.graph.number_of_nodes())


def test_algorithm(autoplan_model, session):
    browser = Browser(session)

    sample_composition = nx.DiGraph()

    edges = [
        ("DTBA_backboneA_splitAMP", "pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1"),
        ("T1MC_NatMX-Cassette_MCT2 (JV)", "pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1"),
        (
            "BBUT_URA3.A.0_homology1_UTP1 (from genome)",
            "pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1",
        ),
        ("MCDT_URA3.A.1_homology2_DTBA", "pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1"),
        ("DH5alpha", "pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1"),
        ("TP-IRES-EGFP-TS", "pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1"),
        (
            "BBUT_URA3.A.1_homology1_UTP1 (from_genome) (new fwd primer))",
            "pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1",
        ),
    ]

    for n1, n2 in edges:
        s1 = browser.find_by_name(n1)
        s2 = browser.find_by_name(n2)
        sample_composition.add_node(s1.id, sample=s1)
        sample_composition.add_node(s2.id, sample=s2)
        sample_composition.add_edge(s1.id, s2.id)

    algorithm = NetworkOptimizer(
        browser, sample_composition, autoplan_model.template_graph
    )
    algorithm.print_sample_composition()

    algorithm.run(session.ObjectType.find_by_name("Plasmid Stock"))


def test_get_sisters_for_run_gel(autoplan_model, session):

    browser = Browser(session)

    sample_composition = nx.DiGraph()

    edges = [
        #     ('DTBA_backboneA_splitAMP', 'pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1'),
        ("T1MC_NatMX-Cassette_MCT2 (JV)", "pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1"),
        (
            "BBUT_URA3.A.0_homology1_UTP1 (from genome)",
            "pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1",
        ),
        ("DH5alpha", "pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1"),
        ("TP-IRES-EGFP-TS", "pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1")
        #     ('MCDT_URA3.A.1_homology2_DTBA', 'pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1'),
        #     ('BBUT_URA3.A.1_homology1_UTP1 (from_genome) (new fwd primer))', 'pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1')
    ]

    for n1, n2 in edges:
        s1 = browser.find_by_name(n1)
        s2 = browser.find_by_name(n2)
        sample_composition.add_node(s1.id, sample=s1)
        sample_composition.add_node(s2.id, sample=s2)
        sample_composition.add_edge(s1.id, s2.id)

    # s = production.Sample.find_by_name('MCDT-LEU-LEU.A.1_homology2_ampR-Split_A')
    # sample_composition.add_node(s.id, sample=s)
    # nx.draw(sample_composition)

    network = NetworkOptimizer(
        browser, sample_composition, autoplan_model.template_graph
    )

    solution = network.run(session.ObjectType.find_by_name("Plasmid Glycerol Stock"))

    run_gels = []
    for n, ndata in solution["graph"].iter_model_data(model_class="AllowableFieldType"):
        aft = ndata["model"]
        if aft.field_type.operation_type.name == "Run Gel":
            if aft.field_type.role == "input":
                run_gels.append((n, ndata))
    run_gels
