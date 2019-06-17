from terrarium.serializer import Serializer
from terrarium.graphs import SampleGraph
from collections import defaultdict


class DataRequester(object):
    """
    The only class that should make ANY requests to aquarium
    """

    def __init__(self, session):
        assert session.using_requests is True
        assert session.using_cache is True
        assert session.browser
        self.session = session

    @classmethod
    def build_sample_graph(cls, samples, g=None, visited=None):
        """
        Requires requests.

        :param samples:
        :type samples:
        :param g:
        :type g:
        :param visited:
        :type visited:
        :return:
        :rtype:
        """
        if visited is None:
            visited = set()
        if g is None:
            g = SampleGraph()

        model_serializer = Serializer.serialize

        sample_data_array = [model_serializer(s) for s in samples]
        sample_data_array = [
            d for d in sample_data_array if g.node_id(d) not in visited
        ]

        if sample_data_array:
            for sample_data in sample_data_array:
                node_id = g.node_id(sample_data)
                visited.add(node_id)
                g.add_data(sample_data)
        else:
            return g

        parent_samples = []
        for sample in samples:
            for fv in sample.field_values:
                if fv.sample:
                    parent_samples.append(fv.sample)
                    m1 = model_serializer(sample)
                    m2 = model_serializer(fv.sample)
                    g.add_edge_from_models(m1, m2)

        browser = sample.session.browser
        browser.get(parent_samples, {"field_values": "sample"})

        return cls.build_sample_graph(parent_samples, g=g, visited=visited)

    # TODO: collect items for data requester
    @classmethod
    def assign_items(cls, graph, browser, sample_ids):
        afts = [ndata for _, ndata in graph.model_data("AllowableFieldType")]
        non_part_afts = [aft for aft in afts if not aft["field_type"]["part"]]
        object_type_ids = list(set([aft["object_type_id"] for aft in non_part_afts]))

        items = browser.where(
            model_class="Item",
            query={"sample_id": sample_ids, "object_type_id": object_type_ids},
        )

        items_by_object_type_id = defaultdict(list)
        for item in items:
            items_by_object_type_id[item.object_type_id].append(item)

        part_by_sample_by_type = cls._find_parts_for_samples(
            browser, sample_ids, lim=50
        )

        new_nodes = []

        for node, ndata in graph.model_data("AllowableFieldType"):
            sample_id = ndata["sample"]["id"]
            if ndata["field_type"]["part"]:
                parts = part_by_sample_by_type.get(ndata["object_type_id"], {}).get(
                    sample_id, []
                )
                for part in parts[-1:]:
                    new_nodes.append(part)

    @staticmethod
    def _find_parts_for_samples(browser, sample_ids, lim=50):
        all_parts = []
        part_type = browser.find_by_name("__Part", model_class="ObjectType")
        for sample_id in sample_ids:
            sample_parts = browser.last(
                lim,
                query=dict(
                    model_class="Item", object_type_id=part_type.id, sample_id=sample_id
                ),
            )
            all_parts += sample_parts
        browser.get(all_parts, "collections")

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
                ).append(Serializer.serialize(part))
        return data
