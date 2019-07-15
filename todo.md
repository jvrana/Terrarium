1. consider using PyMongoDB to store data from 
2. For blueprint creation: First build a directed hypergraph between *any* possible 
AFT and empty instances. Then populate with instances from the plan. Then
use a modular function that computes the cost from the instance.  Return a simple
directed graph. Overall, this will make it easier to add and remove data from
the cost computation (including errors etc.)
3. change .graph to .G
4. `tests/test_algorithms` are hanging
