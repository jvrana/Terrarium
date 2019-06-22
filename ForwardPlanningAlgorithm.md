

## Graph Construction

A directed graph is $G = (V,E)$ constructed from all possible connections

## Experimental planning algorithm using a directed & grouped steiner tree problem:

**Instance** A directed, positively weighted graph:
$$
\begin{gather}
\text{a directed graph} && G = (V, E) \\
\text{with edge cost} && c = E \rightarrow {\rm I\!R^+} \\
\text{a set of starting terminals} && X \subset V \\
\text{a set of ending terminals} && Y \subset V \\
\text{sets (or "groups")} && S_1, ..., S_n \subseteq V \\
\text{} &&S_1 \cap ... \cap S_n = \emptyset \\
\text{sets (or "input groups")} && I_1, ..., I_n \subseteq V \\
\text{} &&I_1 \cap ... \cap I_n = \emptyset \\
\end{gather}
$$
**Solution** A find a directed tree $T = (V_T, E_T)$ in $G$ with minimal **cost function** $\sum_{e \in E_t}{c(e)}$ such that $V_T$ at least one vertex from starting terminals $X$ and every vertex from ending terminals $Y$. Additionally, the tree must contain at least one vertex from every set $S_i$ and for every group $I_i$, either all or none of the vertices in $I_i$ must be included in $V_T$.
$$
\begin{align}
\text{constraint 1} &&& S_i \cap V_T \neq \emptyset \\
\text{constraint 2} &&& I_i \cap V_T \subset \{\emptyset, I_i \} \\
\end{align}
$$


