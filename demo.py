# from autoplanner import AutoPlanner
# from pydent.browser import Browser
# from mysession import production
# import dill
#
# ap = AutoPlanner(production, depth=300)
# ap.construct_template_graph()
# ap.dump('autoplanner.pkl')


from autoplanner import AutoPlanner
from autoplanner.algorithm import NetworkOptimization
from pydent.browser import Browser
from mysession import production
import networkx as nx
%matplotlib inline

ap = AutoPlanner.load('autoplanner.pkl')

ignore_ots = production.OperationType.where({"category": ["Control Blocks", "Library Cloning"], "deployed": True})
ignore_ots.append(production.OperationType.find_by_name("Yeast Mating"))
ignore = [ot.id for ot in ignore_ots]

ap.add_model_filter("AllowableFieldType", lambda m: m.field_type.parent_id in ignore)



browser = Browser(production)

sample_composition = nx.DiGraph()

edges = [
    ('DTBA_backboneA_splitAMP', 'pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1'),
    ('T1MC_NatMX-Cassette_MCT2 (JV)', 'pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1'),
    ('BBUT_URA3.A.0_homology1_UTP1 (from genome)', 'pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1'),
    ('DH5alpha', 'pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1'),
    ('TP-IRES-EGFP-TS', 'pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1' ),
    ('MCDT_URA3.A.1_homology2_DTBA', 'pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1'),
#     ('pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1', 'CEN.PK2 - MAT alpha | his-pGRR-W5-W8-RGR-W36')
    ('BBUT_URA3.A.1_homology1_UTP1 (from_genome) (new fwd primer))', 'pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1')
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

alg = NetworkOptimization(browser, sample_composition, ap.template_graph)
alg.set_verbose(True)
# alg.update_sample_composition
cost, paths, graph = alg.run(production.ObjectType.find_by_name('Plasmid Glycerol Stock'))





from pydent.planner import Planner

canvas = Planner(production)

alg.plan(paths, graph, canvas)

# canvas.draw()
canvas.layout.topo_sort()
canvas.draw()