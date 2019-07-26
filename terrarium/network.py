import math
from collections import OrderedDict
from collections import defaultdict
from functools import reduce
from itertools import count as counter
from itertools import product

import networkx as nx
from pydent.utils.logger import Loggable
from tqdm import tqdm

from terrarium import AutoPlannerModel
from terrarium.browser_graph import BrowserGraph
from terrarium.utils import graph_utils
from terrarium.utils.color_utils import cprint, cstring
from pydent.models import Sample
from pydent.planner import Planner


class NetworkSolution(object):
    def __init__(self, paths: list, graph):
        self.graph = graph
        self.paths = paths


class NetworkFactory(object):
    """Creates a new Terrarium network from a goal and model
    """

    def __init__(self, model):
        """[summary]
        
        :param model: an autoplanner model
        :type model: AutoplannerModel
        """

        self.model = model
        self.algorithms = {}
        self.solution = None

    def add(self, algorithm):
        self.algorithms[algorithm.gid] = algorithm

    def new_from_sample(self, sample):
        scgraph = nx.DiGraph()
        scgraph.add_node(sample.id, sample=sample)
        return self.new_from_composition(scgraph)

    def new_from_composition(self, sample_composition_graph):
        browser = self.model.browser
        template_graph = self.model.template_graph
        algorithm = NetworkOptimizer(browser, sample_composition_graph, template_graph)
        self.add(algorithm)
        return algorithm

    def new_from_edges(self, sample_edges):
        composition = self.sample_composition_from_edges(sample_edges)
        return self.new_from_composition(composition)

    def sample_composition_from_edges(self, sample_edges):
        """
        E.g.

        .. code::

            edges = [
            ('DTBA_backboneA_splitAMP', 'pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1'),
            ('T1MC_NatMX-Cassette_MCT2 (JV)', 'pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1'),
            ('BBUT_URA3.A.0_homology1_UTP1 (from genome)', 'pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1'),
            ('MCDT_URA3.A.1_homology2_DTBA', 'pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1'),
            ('BBUT_URA3.A.1_homology1_UTP1 (from_genome) (new fwd primer))', 'pyMOD-URA-URA3.A.1-pGPD-yeVenus-tCYC1')
        ]
        :param sample_edges:
        :type sample_edges:
        :return:
        :rtype:
        """
        scgraph = nx.DiGraph()

        for n1, n2 in sample_edges:
            s1 = self.model.browser.find_by_name(n1)
            s2 = self.model.browser.find_by_name(n2)
            scgraph.add_node(s1.id, sample=s1)
            scgraph.add_node(s2.id, sample=s2)
            scgraph.add_edge(s1.id, s2.id)
        return scgraph

    @classmethod
    def load_model(cls, path):
        model = AutoPlannerModel.load(path)
        return cls(model)

    # TODO: chaining; incrementally build solution from series of sample_composition graphs
    def chain(self):
        pass


none_sample = Sample()
none_sample.sample_type_id = None
none_sample.id = None
none_sample.name = None


class NetworkOptimizer(Loggable):
    """
    Class that finds optimal Steiner Tree (
    """

    SOLUTION_PATHS = "paths"
    SOLUTION_GRAPH = "graph"
    SOLUTION_COST = "cost"

    counter = counter()

    def __init__(self, browser, sample_composition_graph, template_graph):
        self.browser = browser
        self.sample_composition = sample_composition_graph
        self.template_graph = template_graph.copy()
        self.init_logger("Algorithm")
        self.gid = next(self.counter)

    def _cinfo(self, msg, foreground="white", background="black"):
        self._info(cstring(msg, foreground, background))

    ############################
    # RUN
    ############################

    def run_stage0(self):
        self._cinfo("STAGE 0: Sample Composition")
        self.update_sample_composition()
        graph = self.create_sample_composition_graphs(
            self.template_graph, self.browser, self.sample_composition
        )
        return graph

    def run_stage1(self, graph):
        self._cinfo("STAGE 1 Cleaning Graph")
        self.clean_graph(graph)

    def run_stage2(self, graph, goal_sample, goal_object_type, ignore):
        if ignore is None:
            ignore = []
        self._cinfo("STAGE 2: Terminal Nodes")
        start_nodes = self.extract_items(graph)
        start_nodes += self.extract_leaf_operations(graph)
        start_nodes = [n for n in start_nodes if n not in ignore]
        end_nodes = self.extract_end_nodes(graph, goal_sample, goal_object_type)
        return start_nodes, end_nodes

    def run_stage3(self, graph, start_nodes, end_nodes):
        self._cinfo("STAGE 3: Optimizing")
        return self.optimize_steiner_tree(start_nodes, end_nodes, graph, [])

    def run(self, goal_object_type, goal_sample=None, ignore=None):
        if goal_sample is None:
            goal_sample = self.root_samples()[0]
        if goal_sample.sample_type_id != goal_object_type.sample_type_id:
            raise Exception(
                "ObjectType {} does not match Sample {}".format(
                    goal_object_type.name, goal_sample.name
                )
            )

        ############################
        # Stage 0
        ############################
        graph = self.run_stage0()

        if goal_sample.id not in self.sample_composition:
            raise Exception(
                "Sample id={} not found in sample composition".format(goal_sample.id)
            )

        ############################
        # Stage 1
        ############################
        self.run_stage1(graph)

        ############################
        # Stage 2
        ############################
        start_nodes, end_nodes = self.run_stage2(
            graph, goal_sample, goal_object_type, ignore
        )

        ############################
        # Stage 3
        ############################
        cost, paths, visited_samples = self.run_stage3(graph, start_nodes, end_nodes)

        self.solution = {
            self.SOLUTION_COST: cost,
            self.SOLUTION_PATHS: paths,
            self.SOLUTION_GRAPH: graph,
        }
        return self.solution

    ############################
    # PLAN
    ############################

    def plan(self, canvas=None, solution=None) -> Planner:
        """
        Converts a path through a :class:`BrowserGraph` into an
        Aquarium Plan

        :param paths: list of node_ids
        :param graph: BrowserGraph instance
        :param canvas: Planner instance
        :return:
        """
        if canvas is None:
            canvas = Planner(self.browser.session)
        if solution is None:
            solution = self.solution
        graph = solution[self.SOLUTION_GRAPH].copy()
        paths = solution[self.SOLUTION_PATHS]
        for path_num, path in enumerate(paths):
            print("Path: {}".format(path_num))
            self._plan_assign_field_values(path, graph, canvas)
            self._plan_assign_items(path, graph, canvas)
            print()
        return canvas

    @classmethod
    def _plan_assign_field_values(cls, path, graph, canvas):
        """
        Assign :class:`FieldValue` to a path

        :param path: list of node_ids
        :param graph: BrowserGraph instance
        :param canvas: Planner instance
        :return:
        """
        prev_node = None

        for n, ndata in graph.iter_model_data(
            model_class="AllowableFieldType", nbunch=path
        ):
            aft = ndata["model"]
            sample = ndata["sample"]
            if aft.field_type.role == "output":
                # create and set output operation
                if "operation" not in ndata:
                    print("Creating field value")
                    op = canvas.create_operation_by_type_id(aft.field_type.parent_id)
                    fv = op.output(aft.field_type.name)
                    canvas.set_field_value(fv, sample=sample)
                    ndata["field_value"] = fv
                    ndata["operation"] = op
                else:
                    op = ndata["operation"]

                if prev_node:
                    input_aft = prev_node[1]["model"]
                    input_sample = prev_node[1]["sample"]
                    input_name = input_aft.field_type.name

                    if input_aft.field_type.array:
                        print(
                            "Setting input array {} to sample='{}'".format(
                                input_name, input_sample
                            )
                        )
                        input_fv = canvas.set_input_field_value_array(
                            op, input_name, sample=input_sample
                        )
                    else:
                        print(
                            "Setting input {} to sample='{}'".format(
                                input_name, input_sample
                            )
                        )
                        input_fv = canvas.set_field_value(
                            op.input(input_name), sample=input_sample
                        )
                    print("Setting input field_value for '{}'".format(prev_node[0]))
                    prev_node[1]["field_value"] = input_fv
                    prev_node[1]["operation"] = op
            prev_node = (n, ndata)

    @classmethod
    def _plan_assign_items(cls, path, graph, canvas):
        """
        Assign :class:`Item` in a path

        :param path: list of node_ids
        :param graph: BrowserGraph instance
        :param canvas: Planner instance
        :return:
        """
        prev_node = None

        for n, ndata in graph.iter_model_data(
            model_class="AllowableFieldType", nbunch=path
        ):
            if (
                ndata["node_class"] == "AllowableFieldType"
                and ndata["model"].field_type.role == "input"
            ):
                if "operation" not in ndata:
                    cls.print_aft(graph, n)
                    raise Exception(
                        "Key 'operation' not found in node data '{}'".format(n)
                    )
                input_fv = ndata["field_value"]
                input_sample = ndata["sample"]
                if prev_node:
                    node_class = prev_node[1]["node_class"]
                    if node_class == "AllowableFieldType":
                        output_fv = prev_node[1]["field_value"]
                        canvas.add_wire(output_fv, input_fv)
                    elif node_class == "Item":
                        item = prev_node[1]["model"]
                        if input_fv.field_type.part:
                            canvas.set_part(input_fv, item)
                        else:
                            canvas.set_field_value(
                                input_fv, sample=input_sample, item=item
                            )
                    else:
                        raise Exception(
                            "Node class '{}' not recognized".format(node_class)
                        )
            prev_node = (n, ndata)

    ############################
    # UTILS
    ############################

    def clean_graph(self, graph):
        """
        Remove internal wires with different routing id but same sample

        :param graph:
        :type graph:
        :return:
        :rtype:
        """
        removal = []

        afts = graph.models("AllowableFieldType")
        self.browser.retrieve(afts, "field_type")

        for n1, n2 in tqdm(list(graph.edges)):
            node1 = graph.get_node(n1)
            node2 = graph.get_node(n2)
            if (
                node1["node_class"] == "AllowableFieldType"
                and node2["node_class"] == "AllowableFieldType"
            ):
                aft1 = node1["model"]
                aft2 = node2["model"]
                ft1 = aft1.field_type
                ft2 = aft2.field_type
                if ft1.role == "input" and ft2.role == "output":
                    if (
                        ft1.routing != ft2.routing
                        and node1["sample"].id == node2["sample"].id
                    ):
                        removal.append((n1, n2))
                    if (
                        ft1.routing == ft2.routing
                        and node1["sample"].id != node2["sample"].id
                    ):
                        removal.append((n1, n2))

        print("Removing edges with same sample but different routing ids")
        print(len(graph.edges))
        graph.graph.remove_edges_from(removal)
        return graph

    def update_sample_composition(self):
        updated_sample_composition = self.expand_sample_composition(
            browser=self.browser, graph=self.sample_composition
        )
        self.sample_composition = updated_sample_composition
        return self.sample_composition

    def print_sample_composition(self):
        for s1, s2 in self.sample_composition.edges:
            s1 = self.sample_composition.node[s1]
            s2 = self.sample_composition.node[s2]
            print(s1["sample"].name + " => " + s2["sample"].name)

    def root_samples(self):
        nodes = graph_utils.find_leaves(self.sample_composition)
        return [self.sample_composition.node[n]["sample"] for n in nodes]

    @classmethod
    def expand_sample_composition(cls, browser, samples=None, graph=None):
        if graph is None:
            graph = nx.DiGraph()
        if samples is None:
            graph_copy = nx.DiGraph()
            graph_copy.add_nodes_from(graph.nodes(data=True))
            graph_copy.add_edges_from(graph.edges(data=True))
            graph = graph_copy
            samples = [graph.node[n]["sample"] for n in graph]
        if not samples:
            return graph
        browser.recursive_retrieve(samples, {"field_values": "sample"})
        new_samples = []
        for s1 in samples:
            fvdict = s1._field_value_dictionary()
            for ft in s1.sample_type.field_types:
                if ft.ftype == "sample":
                    fv = fvdict[ft.name]
                    if not isinstance(fv, list):
                        fv = [fv]
                    for _fv in fv:
                        if _fv:
                            s2 = _fv.sample
                            if s2:
                                new_samples.append(s2)
                                graph.add_node(s1.id, sample=s1)
                                graph.add_node(s2.id, sample=s2)
                                graph.add_edge(s2.id, s1.id)
        return cls.expand_sample_composition(browser, new_samples, graph)

    @staticmethod
    def decompose_template_graph_into_samples(
        template_graph, samples, include_none=True
    ):
        """
        From a template graph and list of samples, extract sample specific
        nodes from the template graph (using sample type id)
        :param template_graph:
        :type template_graph:
        :param samples:
        :type samples:
        :return:
        :rtype:
        """
        sample_type_ids = set(s.sample_type_id for s in samples)
        sample_type_graphs = defaultdict(list)

        if include_none:
            sample_type_ids.add(None)
            samples.append(none_sample)

        samples = list(set(samples))

        for n, ndata in template_graph.iter_model_data():
            if ndata["node_class"] == "AllowableFieldType":
                if ndata["model"].sample_type_id in sample_type_ids:
                    sample_type_graphs[ndata["model"].sample_type_id].append(n)

        sample_graphs = {}

        for sample in samples:
            nodes = sample_type_graphs[sample.sample_type_id]
            sample_graph = template_graph.subgraph(nodes)
            sample_graph.set_prefix("Sample{}_".format(sample.id))
            for n, ndata in sample_graph.iter_model_data():
                ndata["sample"] = sample
            sample_graphs[sample.id] = sample_graph
        return sample_graphs

    @staticmethod
    def _find_parts_for_samples(browser, sample_ids, lim=50):
        all_parts = []
        part_type = browser.find_by_name("__Part", model_class="ObjectType")
        for sample_id in sample_ids:
            sample_parts = browser.last(
                lim,
                model_class="Item",
                query=dict(object_type_id=part_type.id, sample_id=sample_id),
            )
            all_parts += sample_parts
        browser.retrieve(all_parts, "collections")

        # filter out parts that do not exist
        all_parts = [
            part
            for part in all_parts
            if part.collections and part.collections[0].location != "deleted"
        ]

        # create a Part-by-Sample-by-ObjectType dictionary
        data = {}
        for part in all_parts:
            if part.collections:
                data.setdefault(part.collections[0].object_type_id, {}).setdefault(
                    part.sample_id, []
                ).append(part)
        return data

    def create_sample_composition_graphs(
        self, template_graph, browser, sample_composition
    ):
        """
        Break a template_graph into subgraphs comprising of individual samples obtained
        from the sample composition graph.

        The `sample_composition` graph is a directed graph that defines how samples may be built
        from other samples. Meanwhile, the `template_graph` defines all possible connections between
        :class:`AllowableFieldType` and the associated weights of each edge determined from the
        :class:`AutoPlannerModel`. Using the `sample_composition` graph, we grab individual subgraphs
        from the template graph for each node in the sample compositions graph (using SampleType).
        The edges of the `sample_composition` graph determines which edges of these new subgraphs
        can be connected to each other, forming the final graph used in the Steiner tree optimization
        algorithm.

        :param template_graph:
        :param browser:
        :param sample_composition:
        :return:
        """
        sample_edges = []
        graphs = []

        samples = []
        for item in sample_composition.nodes:
            samples.append(sample_composition.node[item]["sample"])
        sample_graph_dict = self.decompose_template_graph_into_samples(
            template_graph, samples
        )

        for s1, s2 in sample_composition.edges:
            s1 = sample_composition.node[s1]["sample"]
            s2 = sample_composition.node[s2]["sample"]

            sample_graph1 = sample_graph_dict[s1.id]
            sample_graph2 = sample_graph_dict[s2.id]

            graphs += [sample_graph1.graph, sample_graph2.graph]

            input_afts = [
                aft
                for aft in sample_graph1.iter_models("AllowableFieldType")
                if aft.field_type.role == "input"
            ]
            output_afts = [
                aft
                for aft in sample_graph2.iter_models("AllowableFieldType")
                if aft.field_type.role == "output"
            ]

            pairs = AutoPlannerModel._match_internal_afts(input_afts, output_afts)
            pairs = [
                (sample_graph1.node_id(aft1), sample_graph2.node_id(aft2))
                for aft1, aft2 in pairs
            ]
            sample_edges += pairs

        # include the afts from operations that have no sample_type_id (e.g. Order Primer)
        if None in sample_graph_dict:
            graphs.append(sample_graph_dict[None].graph)
            # TODO: add None edges for internal graphs...

        graph = BrowserGraph(browser)
        graph.graph = nx.compose_all(graphs)

        graph.cache_models()

        self._info("Adding {} sample-to-sample edges".format(len(sample_edges)))

        for n1, n2 in tqdm(sample_edges):
            assert n1 in graph
            assert n2 in graph

            node1 = graph.get_node(n1)
            node2 = graph.get_node(n2)
            aft1 = node1["model"]
            aft2 = node2["model"]

            x = (template_graph.node_id(aft1), template_graph.node_id(aft2))
            edge = template_graph.get_edge(*x)
            graph.add_edge(n1, n2, edge_type="sample_to_sample", weight=edge["weight"])

        afts = list(graph.iter_models(model_class="AllowableFieldType"))
        browser.retrieve(afts, "field_type")
        sample_ids = list(
            set(
                ndata["sample"].id
                for _, ndata in graph.iter_model_data()
                if ndata["sample"] is not None
            )
        )

        ##############################
        # Get items
        ##############################

        non_part_afts = [aft for aft in afts if not aft.field_type.part]
        object_type_ids = list(set([aft.object_type_id for aft in non_part_afts]))

        self._cinfo(
            "finding all relevant items for {} samples and {} object_types".format(
                len(sample_ids), len(object_type_ids)
            )
        )
        items = browser.where(
            model_class="Item",
            query={"sample_id": sample_ids, "object_type_id": object_type_ids},
        )
        items = [item for item in items if item.location != "deleted"]
        self._info("{} total items found".format(len(items)))
        items_by_object_type_id = defaultdict(list)
        for item in items:
            items_by_object_type_id[item.object_type_id].append(item)

        ##############################
        # Get parts
        ##############################

        self._cinfo("finding relevant parts/collections")
        part_by_sample_by_type = self._find_parts_for_samples(
            browser, sample_ids, lim=50
        )
        self._cinfo("found {} collection types".format(len(part_by_sample_by_type)))

        ##############################
        # Assign Items/Parts/Collections
        ##############################

        new_items = []
        new_edges = []
        for node, ndata in graph.iter_model_data(model_class="AllowableFieldType"):
            aft = ndata["model"]
            sample = ndata["sample"]
            if sample:
                sample_id = sample.id
                sample_type_id = sample.sample_type_id
            else:
                sample_id = None
                sample_type_id = None
            if aft.sample_type_id == sample_type_id:
                if aft.field_type.part:
                    parts = part_by_sample_by_type.get(aft.object_type_id, {}).get(
                        sample_id, []
                    )
                    for part in parts[-1:]:
                        if part.sample_id == sample_id:
                            new_items.append(part)
                            new_edges.append((part, sample, node))
                else:
                    items = items_by_object_type_id[aft.object_type_id]
                    for item in items:
                        if item.sample_id == sample_id:
                            new_items.append(item)
                            new_edges.append((item, sample, node))

        for item in new_items:
            graph.add_model(item)

        for item, sample, node in new_edges:
            graph.add_edge(graph.node_id(item), node, weight=0)

        self._info(
            "{} items added to various allowable_field_types".format(len(new_edges))
        )
        return graph

    @staticmethod
    def print_aft(graph, node_id):
        if node_id == "END":
            return
        try:
            node = graph.get_node(node_id)
            if node["node_class"] == "AllowableFieldType":
                aft = node["model"]
                print(
                    "<AFT id={:<10} sample={:<10} {:^10} {:<10} '{:<10}'>".format(
                        aft.id,
                        node["sample"].name,
                        aft.field_type.role,
                        aft.field_type.name,
                        aft.field_type.operation_type.name,
                    )
                )
            elif node["node_class"] == "Item":
                item = node["model"]
                sample_name = "None"
                if item.sample:
                    sample_name = item.sample.name
                print(
                    "<Item id={:<10} {:<20} {:<20}>".format(
                        item.id, sample_name, item.object_type.name
                    )
                )
        except Exception as e:
            print(node_id)
            print(e)
            pass

    def extract_leaf_operations(self, graph):
        """
        Extracts operations that have no inputs (such as "Order Primer")
        :param graph:
        :type graph:
        :return:
        :rtype:
        """

        leaf_afts = []
        for n, ndata in graph.iter_model_data(model_class="AllowableFieldType"):
            aft = ndata["model"]
            if aft.field_type.role == "output":
                node_id = self.template_graph.node_id(aft)
                preds = self.template_graph.predecessors(node_id)
                if not list(preds):
                    leaf_afts.append(n)
        return leaf_afts

    def extract_items(self, graph):
        item_groups = []

        for n, ndata in graph.iter_model_data(model_class="Item"):
            for succ in graph.graph.successors(n):

                grouped = []
                for n2 in graph.graph.predecessors(succ):
                    node = graph.get_node(n2)
                    if node["node_class"] == "Item":
                        grouped.append(n2)
                item_groups.append(tuple(grouped))
        items = list(set(reduce(lambda x, y: list(x) + list(y), item_groups)))

        return items

    def extract_end_nodes(self, graph, goal_sample, goal_object_type):
        end_nodes = []
        for n, ndata in graph.iter_model_data(model_class="AllowableFieldType"):
            aft = ndata["model"]
            if (
                ndata["sample"].id == goal_sample.id
                and aft.object_type_id == goal_object_type.id
                and aft.field_type.role == "output"
            ):
                end_nodes.append(n)
        return end_nodes

    @staticmethod
    def get_sister_inputs(node, node_data, output_node, graph, ignore=None):
        """Returns a field_type_id to nodes"""
        sister_inputs = defaultdict(list)
        if (
            node_data["node_class"] == "AllowableFieldType"
            and node_data["model"].field_type.role == "input"
        ):
            aft = node_data["model"]
            successor = output_node
            predecessors = list(graph.predecessors(successor))
            print(len(predecessors))
            for p in predecessors:
                if p == node or (ignore and p in ignore):
                    continue
                pnode = graph.get_node(p)
                if pnode["node_class"] == "AllowableFieldType":
                    is_array = pnode["model"].field_type.array is True
                    if (
                        not is_array
                        and pnode["model"].field_type_id == aft.field_type_id
                    ):
                        continue
                    if is_array:
                        key = "{}_{}".format(
                            pnode["model"].field_type_id, pnode["sample"].id
                        )
                    else:
                        key = str(pnode["model"].field_type_id)
                    sister_inputs[key].append((p, pnode))
        return sister_inputs

    def _print_nodes(self, node_ids, graph):
        print(node_ids)
        items = list(graph.iter_models(nbunch=node_ids, model_class="Item"))
        self.browser.retrieve(items, "sample")
        self.browser.retrieve(items, "object_type")

        grouped_by_object_type = {}
        for item in items:
            grouped_by_object_type.setdefault(item.object_type.name, []).append(item)

        for otname, items in grouped_by_object_type.items():
            cprint(otname, "white", "black")
            for item in items:
                sample_name = "None"
                if item.sample:
                    sample_name = item.sample.name
                print(
                    "    <Item id={} {} {}".format(
                        item.id, item.object_type.name, sample_name
                    )
                )

        for n, ndata in graph.iter_model_data(
            model_class="AllowableFieldType", nbunch=node_ids
        ):
            self.print_aft(graph, n)

    def _optimize_get_seed_paths(
        self,
        start_nodes,
        end_nodes,
        bgraph,
        visited_end_nodes,
        output_node=None,
        verbose=False,
    ):
        paths = []
        end_nodes = [e for e in end_nodes if e not in visited_end_nodes]
        if verbose:
            print("START")
            self._print_nodes(start_nodes, bgraph)

            print("END")
            print(end_nodes)
            self._print_nodes(end_nodes, bgraph)

            print("VISITED: {}".format(visited_end_nodes))

        for start in start_nodes:
            for end in end_nodes:
                through_nodes = [start, end]
                if output_node:
                    through_nodes.append(output_node)
                try:
                    cost, path = graph_utils.top_paths(through_nodes, bgraph)
                except nx.exception.NetworkXNoPath:
                    #                 print("ERROR: No path from {} to {}".format(start, end))
                    continue
                paths.append((cost, path))
        return paths

    def _gather_assignments(
        self, path, bgraph, visited_end_nodes, visited_samples, depth
    ):

        input_to_output = OrderedDict()
        for n1, n2 in zip(path[:-1], path[1:]):
            node2 = bgraph.get_node(n2)
            node1 = bgraph.get_node(n1)
            if "sample" in node1:
                visited_samples.add(node1["sample"].id)
            if "sample" in node2:
                visited_samples.add(node2["sample"].id)
            if node2["node_class"] == "AllowableFieldType":
                aft2 = node2["model"]
                if aft2.field_type.role == "output":
                    input_to_output[n1] = n2

        print("PATH:")
        for p in path:
            print(p)
            self.print_aft(bgraph, p)

        # iterate through each input to find unfullfilled inputs
        inputs = list(input_to_output.keys())[:]
        print(input_to_output.keys())
        if depth > 0:
            inputs = inputs[:-1]
        #     print()
        #     print("INPUTS: {}".format(inputs))

        #     all_sister
        empty_assignments = defaultdict(list)

        for i, n in enumerate(inputs):
            print()
            print("Finding sisters for:")
            self.print_aft(bgraph, n)
            output_n = input_to_output[n]
            ndata = bgraph.get_node(n)
            sisters = self.get_sister_inputs(
                n, ndata, output_n, bgraph, ignore=visited_end_nodes
            )
            if not sisters:
                print("no sisters found")
            for ftid, nodes in sisters.items():
                print("**Sister FieldType {}**".format(ftid))
                for s, values in nodes:
                    self.print_aft(bgraph, s)
                    empty_assignments["{}_{}".format(output_n, ftid)].append(
                        (s, output_n, values)
                    )
                print()

        ############################################
        # 4.3 recursively find cost & shortest paths
        #     for unassigned inputs for every possible
        #     assignment
        ############################################
        all_assignments = list(product(*empty_assignments.values()))
        print(all_assignments)
        for k, v in empty_assignments.items():
            print(k)
            print(v)

        return all_assignments

    # TODO: fix issue with seed path
    """
    TODO: During the seed stage, this algorithm can get 'stuck' in a non-optimal solution,
    making it difficult to plan 'short' experimental plans. As an example, planning 
    PCRs can get stuck on 'Anneal Oligos' since this is the shortest seed path. But this
    results in a sample penalty since the Template from the sample composition is unfullfilled.
    There is no procedure currently in place to solve this issue.
    
    Solution 1: Instead of using the top seed path, evaluate the top 'N' seed paths, picking the best
    one
    
    Solution 2: Evaluate the top 'N' most 'different' seed paths
    
    Solution 3: Rank seed paths not only on path length/cost, but also on their visited samples.
    The most visited samples, the better the path. However, longer paths have more visited samples,
    hence usually a higher path length/cost. It would be difficult to weight these two
    aspects of the seed path.
    """

    def optimize_steiner_tree(
        self,
        start_nodes,
        end_nodes,
        bgraph,
        visited_end_nodes,
        visited_samples=None,
        output_node=None,
        verbose=True,
        depth=0,
    ):

        # TODO: Algorithm gets stuck on shortest top path...
        # e.g. Yeast Glycerol Stock to Yeast Mating instead of yeast transformation

        if visited_samples is None:
            visited_samples = set()

        ############################################
        # 1. find all shortest paths
        ############################################
        seed_paths = self._optimize_get_seed_paths(
            start_nodes, end_nodes, bgraph, visited_end_nodes, output_node, verbose
        )
        visited_end_nodes += end_nodes

        ############################################
        # 2. find overall shortest path(s)
        ############################################
        NUM_PATHS = 3
        THRESHOLD = 10 ** 8

        if not seed_paths:
            if verbose:
                print("No paths found")
            return math.inf, [], visited_samples
        seed_paths = sorted(seed_paths, key=lambda x: x[0])
        cost, path = seed_paths[0]
        final_paths = [path]
        if cost > THRESHOLD:
            cprint("Path beyond threshold, returning early", "red")
            print(graph_utils.get_path_length(bgraph, path))
            return cost, final_paths, visited_samples

        if verbose:
            cprint("Single path found with cost {}".format(cost), None, "blue")
            cprint(graph_utils.get_path_weights(bgraph, path), None, "blue")

        ############################################
        # 3. mark edges as 'visited'
        ############################################
        bgraph_copy = bgraph.copy()
        edges = list(zip(path[:-1], path[1:]))
        for e1, e2 in edges:
            edge = bgraph_copy.get_edge(e1, e2)
            edge["weight"] = 0

        ############################################
        # 4.1 input-to-output graph
        ############################################
        input_to_output = OrderedDict()
        for n1, n2 in zip(path[:-1], path[1:]):
            node2 = bgraph_copy.get_node(n2)
            node1 = bgraph_copy.get_node(n1)
            if "sample" in node1:
                visited_samples.add(node1["sample"].id)
            if "sample" in node2:
                visited_samples.add(node2["sample"].id)
            if node2["node_class"] == "AllowableFieldType":
                aft2 = node2["model"]
                if aft2.field_type.role == "output":
                    input_to_output[n1] = n2

        ############################################
        # 4.2  search for all unassigned inputs
        ############################################
        print("PATH:")
        for p in path:
            print(p)
            self.print_aft(bgraph, p)

        # iterate through each input to find unfullfilled inputs
        inputs = list(input_to_output.keys())[:]
        print(input_to_output.keys())
        if depth > 0:
            inputs = inputs[:-1]
        #     print()
        #     print("INPUTS: {}".format(inputs))

        #     all_sister
        empty_assignments = defaultdict(list)

        for i, n in enumerate(inputs):
            print()
            print("Finding sisters for:")
            self.print_aft(bgraph, n)
            output_n = input_to_output[n]
            ndata = bgraph_copy.get_node(n)
            sisters = self.get_sister_inputs(
                n, ndata, output_n, bgraph_copy, ignore=visited_end_nodes
            )
            if not sisters:
                print("no sisters found")
            for ftid, nodes in sisters.items():
                print("**Sister FieldType {}**".format(ftid))
                for s, values in nodes:
                    self.print_aft(bgraph, s)
                    empty_assignments["{}_{}".format(output_n, ftid)].append(
                        (s, output_n, values)
                    )
                print()

        ############################################
        # 4.3 recursively find cost & shortest paths
        #     for unassigned inputs for every possible
        #     assignment
        ############################################
        all_assignments = list(product(*empty_assignments.values()))
        print(all_assignments)
        for k, v in empty_assignments.items():
            print(k)
            print(v)
        if all_assignments[0]:

            # TODO: enforce unique sample_ids if in operation_type
            cprint("Found {} assignments".format(len(all_assignments)), None, "blue")
            best_assignment_costs = []

            for assign_num, assignment in enumerate(all_assignments):
                cprint(
                    "Evaluating assignment {}/{}".format(
                        assign_num + 1, len(all_assignments)
                    ),
                    None,
                    "red",
                )
                cprint("Assignment length: {}".format(len(assignment)), None, "yellow")

                assignment_cost = 0
                assignment_paths = []
                assignment_samples = set(visited_samples)
                for end_node, output_node, _ in assignment:
                    _cost, _paths, _visited_samples = self.optimize_steiner_tree(
                        start_nodes,
                        [end_node],
                        bgraph_copy,
                        visited_end_nodes[:],
                        assignment_samples,
                        output_node,
                        verbose=True,
                        depth=depth + 1,
                    )
                    assignment_cost += _cost
                    assignment_paths += _paths
                    assignment_samples = assignment_samples.union(_visited_samples)
                best_assignment_costs.append(
                    (assignment_cost, assignment_paths, assignment_samples)
                )
            cprint([(len(x[2]), x[0]) for x in best_assignment_costs], "green")
            best_assignment_costs = sorted(
                best_assignment_costs, key=lambda x: (-len(x[2]), x[0])
            )

            cprint(
                "Best assignment cost returned: {}".format(best_assignment_costs[0][0]),
                "red",
            )

            cost += best_assignment_costs[0][0]
            final_paths += best_assignment_costs[0][1]
            visited_samples = visited_samples.union(best_assignment_costs[0][2])

        ############################################
        # 5 Make a sample penalty for missing input samples
        ############################################

        output_samples = set()
        for path in final_paths:
            for node in path:
                ndata = bgraph_copy.get_node(node)
                if "sample" in ndata:
                    output_samples.add(ndata["sample"])

        expected_samples = set()
        for sample in output_samples:
            for pred in self.sample_composition.predecessors(sample.id):
                expected_samples.add(pred)

        ############################################
        # return cost and paths
        ############################################

        sample_penalty = max(
            [(len(expected_samples) - len(visited_samples)) * 10000, 0]
        )
        cprint("SAMPLES {}/{}".format(len(visited_samples), len(expected_samples)))
        cprint("COST AT DEPTH {}: {}".format(depth, cost), None, "red")
        cprint("SAMPLE PENALTY: {}".format(sample_penalty))
        cprint("VISITED SAMPLES: {}".format(visited_samples), None, "red")
        return cost + sample_penalty, final_paths, visited_samples
