import networkx as nx
from pydent.browser import Browser

from autoplanner.algorithm import Algorithm


def test_algorithm(autoplanner, session):
    browser = Browser(session)

    sample_composition = nx.DiGraph()

    edges = [
        ('DTBA_backboneA_splitAMP', 'pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1'),
        ('T1MC_NatMX-Cassette_MCT2 (JV)', 'pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1'),
        ('BBUT_URA3.A.0_homology1_UTP1 (from genome)', 'pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1'),
        ('MCDT_URA3.A.1_homology2_DTBA', 'pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1'),
        ('BBUT_URA3.A.1_homology1_UTP1 (from_genome) (new fwd primer))', 'pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1')
    ]

    for n1, n2 in edges:
        s1 = browser.find_by_name(n1)
        s2 = browser.find_by_name(n2)
        sample_composition.add_node(s1.id, sample=s1)
        sample_composition.add_node(s2.id, sample=s2)
        sample_composition.add_edge(s1.id, s2.id)

    algorithm = Algorithm(browser, sample_composition, autoplanner.template_graph)
    algorithm.print_sample_composition()

    algorithm.run(session.ObjectType.find_by_name('Plasmid Stock'))
