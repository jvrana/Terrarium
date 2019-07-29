# Terrarium

[![PyPI version](https://badge.fury.io/py/terrarium-capp.svg)](https://badge.fury.io/py/terrarium-capp)

This piece of software automatically plans scientific experiments in Aquarium using historical
planning data and current laboratory inventory. Data can be pulled from specific researchers
to emulate how that particular researcher would plan experiments.

## Requirements

* development version of **trident (v0.1.0)**
* Python >= 3.6
* Aquarium login credentials

## Usage

Installing a specific version

```python
pip install terrarium-capp==0.1.2
```

New models can be built as in the following:

```python
from pydent import AqSession
from terrarium import AutoPlannerModel
production = AqSession("login", "pass", "url")

# pull last 300 experimental to build model
model = AutoPlannerModel(production, depth=300)
model.build()
models.save('terrarium.pkl')
```

Saved models can be open later:

```python
model = AutoPlannerModel.load('terrarium.pkl')
```

What protocols the model uses can be adjusted using filters:

```python
ignore_ots = production.OperationType.where({"category": ["Control Blocks", "Library Cloning"], "deployed": True})
ignore_ots += production.OperationType.where({"name": "Yeast Mating"})
ignore_ots += production.OperationType.where({"name": "Yeast Auxotrophic Plate Mating"})
ignore = [ot.id for ot in ignore_ots]
model.add_model_filter("AllowableFieldType", lambda m: m.field_type.parent_id in ignore)
```

Sample composition:

```python
sample_composition = nx.DiGraph()

# build a new yeast strain from a plasmid, which is comprised of several fragments
edges = [
     ('DTBA_backboneA_splitAMP', 'pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1'),
     ('T1MC_NatMX-Cassette_MCT2 (JV)', 'pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1'),
     ('BBUT_URA3.A.0_homology1_UTP1 (from genome)', 'pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1'),
     ('DH5alpha', 'pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1'),
     ('TP-IRES-EGFP-TS', 'pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1' ),
     ('pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1', 'CEN.PK2 - MAT alpha | his-pGRR-W5-W8-RGR-W36'),
]

for n1, n2 in edges:
    s1 = browser.find_by_name(n1)
    s2 = browser.find_by_name(n2)
    sample_composition.add_node(s1.id, sample=s1)
    sample_composition.add_node(s2.id, sample=s2)
    sample_composition.add_edge(s1.id, s2.id)
```

```python
ignore_items = []  # optional to not include certain items in the search.
desired_object_type = production.ObjectType.find_by_name('Fragment Stock')
cost, paths, graph = network.run(desired_object_type, ignore=ignore_items)
```

```python
# make a new plan
canvas = Planner(production)

# add protocols from optimized network to plan
network.plan(paths, graph, canvas)

# submit to Aquarium
canvas.create()
```

**Example of Planning Yeast Construction**

![plan_example](assets/images/plan_example0.png)

**Probability Matrix of Connecting Aquarium Protocols**

The autoplanner uses this type of data, in concert with the `sample_composition` network,
to build an optimal experiment.

![all_connections](assets/images/all_op_types.png)

**Top 50 Connections**

![top_50_connections](assets/images/top_50_optypes.png)

## Model Factory

```python
factory = ModelFactory(session)

# make a model from a single user
model1 = factory.emulate('user1').build()

# make a model from a group of users
user_group = ['user2', 'user3']
model2 = factory.emulate(user_group).build()

# make a model from the last 100 plans
model3 = factory.new(100).build()

# compose a weighted model
model = model1 + model2 * 3
```

## Future Version

* estimate convidence for certain inventory items or operations based on
past success rate
* better api for
* using 'ghost' plans to build model
* emulating specific users / user groups
** faster execution (currently ~45-60 seconds)

## License

Feb. 4, 2019 - This software is not currently licensed. The author (Justin D. Vrana of University of Washington) does not grant permission to copy or modify code base.
