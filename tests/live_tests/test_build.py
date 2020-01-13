from os.path import join

import networkx as nx
from pydent.planner import Planner

from terrarium import AutoPlannerModel
from terrarium import NetworkFactory


def test_build(session, tmp_path):
    # pull last 10 experimental to build model
    with session.with_cache(timeout=60) as sess:
        model = AutoPlannerModel(sess.browser, plans=sess.Plan.last(10))
        model.build()
        model.save(join(tmp_path, "tmpmodel.pkl"))


def test_build_and_plan(session):
    # pull last 10 experimental to build model
    with session.with_cache(timeout=60) as sess:
        model = AutoPlannerModel(sess.browser, plans=sess.Plan.last(10))
        model.build()

        ignore_ots = sess.OperationType.where(
            {"category": ["Control Blocks", "Library Cloning"], "deployed": True}
        )
        ignore_ots += sess.OperationType.where({"name": "Yeast Mating"})
        ignore_ots += sess.OperationType.where(
            {"name": "Yeast Auxotrophic Plate Mating"}
        )
        ignore = [ot.id for ot in ignore_ots]
        model.add_model_filter(
            "AllowableFieldType",
            AutoPlannerModel.EXCLUDE_FILTER,
            lambda m: m.field_type.parent_id in ignore,
        )

        edges = [
            ("DTBA_backboneA_splitAMP", "pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1"),
            ("T1MC_NatMX-Cassette_MCT2 (JV)", "pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1"),
            (
                "BBUT_URA3.A.0_homology1_UTP1 (from genome)",
                "pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1",
            ),
            ("DH5alpha", "pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1"),
            ("TP-IRES-EGFP-TS", "pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1"),
            (
                "pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1",
                "CEN.PK2 - MAT alpha | his-pGRR-W5-W8-RGR-W36",
            ),
        ]
        factory = NetworkFactory(model)
        network = factory.new_from_edges(edges)

        ignore_items = []  # optional to not include certain items in the search.
        desired_object_type = sess.ObjectType.find_by_name("Yeast Glycerol Stock")

        network.run(desired_object_type, ignore=ignore_items)
        print("template graph")
        print(nx.info(model.template_graph.graph))

        print("graph")
        print(nx.info(network.solution.graph.graph))
        plan = network.plan()
