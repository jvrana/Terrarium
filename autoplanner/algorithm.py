import networkx as nx
from collections import defaultdict
from autoplanner import AutoPlanner
from autoplanner.autoplanner import BrowserGraph
from functools import reduce
import math
from autoplanner.utils.color_utils import cprint, cstring
from autoplanner.utils import graph_utils
from itertools import product
from collections import OrderedDict

from tqdm import tqdm
from itertools import count as counter

from pydent.utils.logger import Loggable


class AlgorithmFactory(object):

    def __init__(self, browser, template_graph):
        self.browser = browser
        self.template_graph = template_graph
        self.algorithms = {}

    def add(self, algorithm):
        self.algorithms[algorithm.gid] = algorithm

    def new_from_composition(self, sample_composition_graph):
        algorithm = Algorithm(self.browser, sample_composition_graph, self.template_graph.copy())
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
            s1 = self.browser.find_by_name(n1)
            s2 = self.browser.find_by_name(n2)
            scgraph.add_node(s1.id, sample=s1)
            scgraph.add_node(s2.id, sample=s2)
            scgraph.add_edge(s1.id, s2.id)
        return scgraph


class Algorithm(Loggable):

    counter = counter()

    def __init__(self, browser, sample_composition_graph, template_graph):
        self.browser = browser
        self.sample_composition = sample_composition_graph
        self.template_graph = template_graph.copy()
        self.init_logger("Algorithm")
        self.gid = next(self.counter)

    def _cinfo(self, msg, foreground='white', background='black'):
        self._info(cstring(msg, foreground, background))

    def stage0(self):
        self._cinfo("STAGE 0: Sample Composition")
        self.update_sample_composition()
        graph = self.create_sample_composition_graphs(self.template_graph, self.browser, self.sample_composition)
        return graph

    def stage1(self, graph):
        self._cinfo("STAGE 1 Cleaning Graph")
        self.clean_graph(graph)

    def stage2(self, graph, goal_sample, goal_object_type, ignore):
        if ignore is None:
            ignore = []
        self._cinfo("STAGE 2: Terminal Nodes")
        start_nodes = self.extract_items(graph)
        start_nodes = [n for n in start_nodes if n not in ignore]
        end_nodes = self.extract_end_nodes(graph, goal_sample, goal_object_type)
        return start_nodes, end_nodes

    def stage3(self, graph, start_nodes, end_nodes):
        self._cinfo("STAGE 3: Optimizing")
        return self.optimize_steiner_tree(start_nodes, end_nodes, graph, [])

    def run(self, goal_object_type, ignore=None):
        goal_sample = self.goal_samples()[0]
        if goal_sample.sample_type_id != goal_object_type.sample_type_id:
            raise Exception("ObjectType {} does not match Sample {}".format(goal_object_type.name, goal_sample.name))

        ############################
        # Stage 0
        ############################
        graph = self.stage0()

        ############################
        # Stage 1
        ############################
        self.stage1(graph)

        ############################
        # Stage 2
        ############################
        start_nodes, end_nodes = self.stage2(graph, goal_sample, goal_object_type, ignore)

        ############################
        # Stage 3
        ############################
        cost, paths = self.stage3(graph, start_nodes, end_nodes)

        return cost, paths

    @classmethod
    def plan(cls, paths, graph, canvas):
        for path_num, path in enumerate(paths):
            print("Path: {}".format(path_num))
            cls.path_to_operation_chain(path, graph, canvas)
            print()
        return canvas

    @classmethod
    def path_to_operation_chain(cls, path, graph, canvas):
        nodes = [(n, graph.get_node(n)) for n in path]
        print("Path:")
        for p in path:
            print(p)
        print("Creating Operations")
        prev_node = None
        for n, ndata in nodes:
            if ndata['node_class'] == 'AllowableFieldType':
                aft = ndata['model']
                if aft.field_type.role == 'output':
                    # create and set output operation and output field_value
                    if 'operation' not in ndata:
                        op = canvas.create_operation_by_id(aft.field_type.parent_id)
                        fv = canvas.set_field_value(op.output(aft.field_type.name), sample=ndata['sample'])
                        print("Creating output operation for '{}'".format(n))
                        ndata['field_value'] = fv
                        ndata['operation'] = op
                    else:
                        op = ndata['operation']
                        fv = ndata['field_value']

                    # set input
                    input_aft = prev_node[1]['model']
                    input_sample = prev_node[1]['sample']
                    input_name = input_aft.field_type.name
                    if input_aft.field_type.array:
                        print("Adding to array")
                        input_fv = canvas.set_input_field_value_array(op, input_name, sample=input_sample)
                    else:
                        input_fv = canvas.set_field_value(op.input(input_name), sample=input_sample)
                    if prev_node:
                        print("Setting input field_value for '{}'".format(prev_node[0]))
                        prev_node[1]['field_value'] = input_fv
                        prev_node[1]['operation'] = op
            prev_node = (n, ndata)

        prev_node = None
        for n, ndata in nodes:
            if ndata['node_class'] == 'AllowableFieldType' and ndata['model'].field_type.role == 'input':
                aft = ndata['model']
                if 'operation' not in ndata:
                    cls.print_aft(graph, n)
                    raise Exception("Key 'operation' not found in node data '{}'".format(n))
                input_op = ndata['operation']
                input_fv = ndata['field_value']
                input_sample = ndata['sample']
                node_class = prev_node[1]['node_class']
                if node_class == 'AllowableFieldType':
                    output_fv = prev_node[1]['field_value']
                    canvas.add_wire(output_fv, input_fv)
                elif node_class == 'Item':
                    item = prev_node[1]['model']
                    canvas.set_field_value(input_fv, sample=input_sample, item=item)
                else:
                    raise Exception("Node class '{}' not recognized".format(node_class))
            prev_node = (n, ndata)


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
        self.browser.retrieve(afts, 'field_type')

        for n1, n2 in tqdm(list(graph.edges)):
            node1 = graph.get_node(n1)
            node2 = graph.get_node(n2)
            if node1['node_class'] == 'AllowableFieldType' and node2['node_class'] == 'AllowableFieldType':
                aft1 = node1['model']
                aft2 = node2['model']
                ft1 = aft1.field_type
                ft2 = aft2.field_type
                if ft1.role == 'input' and ft2.role == 'output':
                    if ft1.routing != ft2.routing and node1['sample'].id == node2['sample'].id:
                        removal.append((n1, n2))
                    if ft1.routing == ft2.routing and node1['sample'].id != node2['sample'].id:
                        removal.append((n1, n2))

        print("Removing edges with same sample but different routing ids")
        print(len(graph.edges))
        graph.graph.remove_edges_from(removal)
        return graph

    def update_sample_composition(self):
        updated_sample_composition = self.build_sample_graph(graph=self.sample_composition)
        self.sample_composition = updated_sample_composition

    def print_sample_composition(self):
        for s1, s2 in self.sample_composition.edges:
            s1 = self.sample_composition.node[s1]
            s2 = self.sample_composition.node[s2]
            print(s1['sample'].name + " => " + s2['sample'].name)

    def goal_samples(self):
        nodes = graph_utils.find_leaves(self.sample_composition)
        return [self.sample_composition.node[n]['sample'] for n in nodes]

    def build_sample_graph(self, samples=None, graph=None):
        if graph is None:
            graph = nx.DiGraph()
        if samples is None:
            graph_copy = nx.DiGraph()
            graph_copy.add_nodes_from(graph.nodes(data=True))
            graph_copy.add_edges_from(graph.edges(data=True))
            graph = graph_copy
            samples = [graph.node[n]['sample'] for n in graph]
        if not samples:
            return graph
        self.browser.recursive_retrieve(samples, {'field_values': 'sample'})
        new_samples = []
        for s1 in samples:
            if True: #s1.id not in graph:
                for fv in s1.field_values:
                    if fv.sample:
                        s2 = fv.sample
                        new_samples.append(s2)
                        graph.add_node(s1.id, sample=s1)
                        graph.add_node(s2.id, sample=s2)
                        graph.add_edge(s2.id, s1.id)
        return self.build_sample_graph(new_samples, graph)

    @staticmethod
    def decompose_template_graph_into_samples(template_graph, samples):
        sample_type_ids = set(s.sample_type_id for s in samples)
        sample_type_graphs = defaultdict(list)

        for n, ndata in template_graph.iter_model_data():
            if ndata['node_class'] == 'AllowableFieldType':
                if ndata['model'].sample_type_id in sample_type_ids:
                    sample_type_graphs[ndata['model'].sample_type_id].append(n)

        sample_graphs = {}

        for sample in samples:
            nodes = sample_type_graphs[sample.sample_type_id]
            sample_graph = template_graph.subgraph(nodes)
            sample_graph.set_prefix("Sample{}_".format(sample.id))
            for n, ndata in sample_graph.iter_model_data():
                ndata['sample'] = sample
            sample_graphs[sample.id] = sample_graph
        return sample_graphs

    def create_sample_composition_graphs(self, template_graph, browser, sample_composition):
        sample_edges = []
        graphs = []

        samples = []
        for n in sample_composition.nodes:
            samples.append(sample_composition.node[n]['sample'])
        sample_graph_dict = self.decompose_template_graph_into_samples(template_graph, samples)

        for s1, s2 in sample_composition.edges:
            s1 = sample_composition.node[s1]['sample']
            s2 = sample_composition.node[s2]['sample']

            sample_graph1 = sample_graph_dict[s1.id]
            sample_graph2 = sample_graph_dict[s2.id]

            graphs += [sample_graph1.graph, sample_graph2.graph]

            input_afts = [aft for aft in sample_graph1.iter_models("AllowableFieldType") if
                          aft.field_type.role == 'input']
            output_afts = [aft for aft in sample_graph2.iter_models("AllowableFieldType") if
                           aft.field_type.role == 'output']

            pairs = AutoPlanner.match_internal_afts(input_afts, output_afts)
            pairs = [
                (sample_graph1.node_id(aft1), sample_graph2.node_id(aft2)) for aft1, aft2 in pairs
            ]
            sample_edges += pairs

        graph = BrowserGraph(browser)
        graph.graph = nx.compose_all(graphs)

        graph.cache_models()

        self._info("Adding {} sample-to-sample edges".format(len(sample_edges)))

        for n1, n2 in tqdm(sample_edges):
            assert n1 in graph
            assert n2 in graph

            node1 = graph.get_node(n1)
            node2 = graph.get_node(n2)
            aft1 = node1['model']
            aft2 = node2['model']

            x = (template_graph.node_id(aft1), template_graph.node_id(aft2))
            edge = template_graph.get_edge(*x)
            graph.add_edge(n1, n2, edge_type="sample_to_sample", weight=edge['weight'])

        afts = list(graph.iter_models(model_class="AllowableFieldType"))
        object_type_ids = list(set([aft.object_type_id for aft in afts]))
        sample_ids = list(set(ndata['sample'].id for _, ndata in graph.iter_model_data()))

        self._info("finding items...")
        items = browser.where(model_class="Item", query={'sample_id': sample_ids, 'object_type_id': object_type_ids})
        items = [item for item in items if item.location != 'deleted']
        self._info("{} total items found".format(len(items)))
        items_by_object_type_id = defaultdict(list)
        for item in items:
            items_by_object_type_id[item.object_type_id].append(item)

        new_nodes = []
        new_edges = []
        for node, ndata in graph.iter_model_data(model_class='AllowableFieldType'):
            aft = ndata['model']
            sample = ndata['sample']
            if True:  # aft.field_type.role == 'input':
                items = items_by_object_type_id[aft.object_type_id]
                for item in items:
                    if item.sample_id == sample.id and sample.sample_type_id == aft.sample_type_id:
                        new_nodes.append(item)
                        new_edges.append((item, sample, node))

        for n in new_nodes:
            graph.add_node(n)

        for item, sample, node in new_edges:
            graph.add_edge(graph.node_id(item), node, weight=0)

        self._info("{} items added to various allowable_field_types".format(len(new_edges)))
        return graph

    @staticmethod
    def print_aft(graph, node_id):
        if node_id == 'END':
            return
        try:
            node = graph.get_node(node_id)
            if node['node_class'] == 'AllowableFieldType':
                aft = node['model']
                print("<AFT id={:<10} sample={:<10} {:^10} {:<10} '{:<10}'>".format(
                    aft.id,
                    node['sample'].name,
                    aft.field_type.role,
                    aft.field_type.name,
                    aft.field_type.operation_type.name
                ))
            elif node['node_class'] == 'Item':
                item = node['model']
                print("<Item id={:<10} {:<20} {:<20}>".format(
                    item.id,
                    item.sample.name,
                    item.object_type.name,
                ))
        except:
            pass

    def extract_items(self, graph):
        item_groups = []

        for n, ndata in graph.iter_model_data(model_class="Item"):
            for succ in graph.graph.successors(n):

                grouped = []
                for n2 in graph.graph.predecessors(succ):
                    node = graph.get_node(n2)
                    if node['node_class'] == 'Item':
                        grouped.append(n2)
                item_groups.append(tuple(grouped))
        items = list(set(reduce(lambda x, y: list(x) + list(y), item_groups)))
        return items

    def extract_end_nodes(self, graph, goal_sample, goal_object_type):
        end_nodes = []
        for n, ndata in graph.iter_model_data(model_class="AllowableFieldType"):
            aft = ndata['model']
            if ndata['sample'].id == goal_sample.id and aft.object_type_id == goal_object_type.id and aft.field_type.role == 'output':
                end_nodes.append(n)
        return end_nodes

    @staticmethod
    def get_sister_inputs(node, node_data, output_node, graph, ignore=None):
        """Returns a field_type_id to nodes"""
        sister_inputs = defaultdict(list)
        if node_data['node_class'] == 'AllowableFieldType' and node_data['model'].field_type.role == 'input':
            aft = node_data['model']
            successor = output_node
            predecessors = list(graph.predecessors(successor))
            for p in predecessors:
                if p == node or (ignore and p in ignore):
                    continue
                pnode = graph.get_node(p)
                if pnode['node_class'] == 'AllowableFieldType':
                    is_array = pnode['model'].field_type.array == True
                    if not is_array and pnode['model'].field_type_id == aft.field_type_id:
                        #                     print("is not array and field_type_id same")
                        continue
                    # TODO: maybe uncomment this? idk
                    #                 if pnode['sample'].id == node_data['sample'].id:
                    #                     continue
                    if is_array:
                        key = '{}_{}'.format(pnode['model'].field_type_id, pnode['sample'].id)
                    else:
                        key = str(pnode['model'].field_type_id)
                    sister_inputs[key].append((p, pnode))
        return sister_inputs

    @classmethod
    def optimize_steiner_tree(cls, start_nodes, end_nodes, bgraph, visited_end_nodes, visited_samples=None, output_node=None,
                              progress_bar=False, verbose=False, depth=0):

        # TODO: Add output item to end nodes

        if visited_samples is None:
            visited_samples = set()

        ############################################
        # 1. find all shortest paths
        ############################################



        paths = []
        end_nodes = [e for e in end_nodes if e not in visited_end_nodes]
        if verbose:
            print(depth)
            print("START: {}".format(start_nodes))
            print("END: {}".format(end_nodes))
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

        # do not visit end nodes again
        visited_end_nodes += end_nodes

        ############################################
        # 2. find overall shortest path
        ############################################
        if not paths:
            if verbose:
                print("No paths found")
            return math.inf, []
        paths = sorted(paths, key=lambda x: x[0])
        cost, path = paths[0]
        final_paths = [path]

        if verbose:
            cprint("Single path found with cost {}".format(cost), None, 'blue')

        ############################################
        # 3. mark edges as 'visited'
        ############################################
        bgraph_copy = bgraph.copy()
        edges = list(zip(path[:-1], path[1:]))
        for e1, e2 in edges:
            edge = bgraph_copy.get_edge(e1, e2)
            edge['weight'] = 0

        ############################################
        # 4.1 input-to-output graph
        ############################################
        input_to_output = OrderedDict()
        for n1, n2 in zip(path[:-1], path[1:]):
            node2 = bgraph_copy.get_node(n2)
            node1 = bgraph_copy.get_node(n1)
            if 'sample' in node1:
                visited_samples.add(node1['sample'].id)
            if 'sample' in node2:
                visited_samples.add(node2['sample'].id)
            if node2['node_class'] == 'AllowableFieldType':
                aft2 = node2['model']
                if aft2.field_type.role == 'output':
                    input_to_output[n1] = n2

        ############################################
        # 4.2  search for all unassigned inputs
        ############################################
        print("PATH:")
        for p in path:
            print(p)
            cls.print_aft(bgraph, p)

        # iterate through each input to find unfullfilled inputs
        inputs = list(input_to_output.keys())[:]
        if depth > 0:
            inputs = inputs[1:]
        #     print()
        #     print("INPUTS: {}".format(inputs))

        #     all_sister
        empty_assignments = defaultdict(list)

        for i, n in enumerate(inputs):
            print("Finding sisters for:")
            cls.print_aft(bgraph, n)
            output_n = input_to_output[n]
            ndata = bgraph_copy.get_node(n)
            sisters = cls.get_sister_inputs(n, ndata, output_n, bgraph_copy, ignore=visited_end_nodes)
            for ftid, nodes in sisters.items():
                print("**Sister FieldType {}**".format(ftid))
                for s, values in nodes:
                    cls.print_aft(bgraph, s)
                    empty_assignments['{}_{}'.format(output_n, s)].append((s, output_n, values))
                print()

        ############################################
        # 4.3 recursively find cost & shortest paths
        #     for unassigned inputs for every possible
        #     assignment
        ############################################
        all_assignments = list(product(*empty_assignments.values()))
        if all_assignments[0]:

            # TODO: enforce unique sample_ids if in operation_type
            cprint("Found {} assignments".format(len(all_assignments)), None, 'blue')

            best_assignment_costs = []

            for assign_num, assignment in enumerate(all_assignments):
                cprint("Evaluating assignment {}/{}".format(assign_num + 1, len(all_assignments)), None, 'red')
                cprint("Assignment length: {}".format(len(assignment)), None, 'yellow')

                assignment_cost = 0
                assignment_paths = []
                for end_node, output_node, _ in assignment:
                    _cost, _paths = cls.optimize_steiner_tree(start_nodes, [end_node], bgraph_copy, visited_end_nodes[:],
                                                          visited_samples, output_node, verbose=True, depth=depth + 1)
                    assignment_cost += _cost
                    assignment_paths += _paths

                best_assignment_costs.append((assignment_cost, assignment_paths))

            best_assignment_costs = sorted(best_assignment_costs)

            cost += best_assignment_costs[0][0]
            final_paths += best_assignment_costs[0][1]

        ############################################
        # 5 return cost and paths
        ############################################
        cprint("COST AT DEPTH {}: {}".format(depth, cost), None, 'red')
        cprint("VISITED SAMPLES: {}".format(visited_samples), None, 'red')
        return cost, final_paths
