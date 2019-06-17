from terrarium.serializer import Serializer
from terrarium.graphs import SampleGraph
from collections import defaultdict
from typing import Sequence


class DataRequester(object):
    """
    The only class that should make ANY requests to aquarium
    """

    def __init__(self, session):
        assert session.using_requests is True
        assert session.using_cache is True
        assert session.browser
        self.session = session

    @property
    def browser(self):
        return self.session.browser

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

    def collect_items(
        self, afts: Sequence[dict], sample_ids: Sequence[int]
    ) -> Sequence[dict]:
        """

        :param sample_ids: list of serialized allowable field types containing keys ['field_type']['part']
        :type sample_ids: list
        :param sample_ids: list of sample ids
        :type sample_ids: list
        :return: list of serialized items
        :rtype: list
        """
        non_part_afts = [aft for aft in afts if not aft["field_type"]["part"]]
        object_type_ids = list(set([aft["object_type_id"] for aft in non_part_afts]))

        items = self.session.Item.where(
            {"sample_id": sample_ids, "object_type_id": object_type_ids}
        )
        return [Serializer.serialize(i) for i in items]

    def collect_parts(self, sample_ids, lim):
        """

        :param sample_ids: list of sample ids
        :type sample_ids: list
        :param lim: maximum number of parts to return per sample
        :type lim: int
        :return: list of serialized items
        :rtype: list
        """
        parts_list = []
        part_type = self.session.ObjectType.find_by_name("__Part")
        for sample_id in sample_ids:
            parts_list += self.session.Item.last(
                lim, query={"object_type_id": part_type.id, "sample_id": sample_id}
            )

        self.session.browser.get(parts_list, "collections")
        parts_list = [
            part
            for part in parts_list
            if part.collections and part.collections[0].location != "deleted"
        ]

        return [Serializer.serialize(part, include='collections') for part in parts_list]

    def collect_afts_from_plans(self, num):
        plans = self.session.Plan.last(num)
        nodes, edges = Serializer.serialize_plans(plans)
        return nodes, edges

    def collect_deployed_afts(self):
        with self.session.with_cache(timeout=60) as sess:
            sess.OperationType.where({"deployed": True})

            sess.browser.get(
                "OperationType",
                {
                    "field_types": {
                        "allowable_field_types": {
                            "object_type",
                            "sample_type",
                            "field_type",
                        }
                    }
                },
            )

            afts = sess.browser.get("AllowableFieldType")
        return [Serializer.serialize_aft(aft) for aft in afts]

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

    def find_parts(self, sample_ids, lim=50):
        all_parts = []
        part_type = self.browser.find_by_name("__Part", model_class="ObjectType")
        for sample_id in sample_ids:
            sample_parts = self.browser.last(
                lim,
                query=dict(
                    model_class="Item", object_type_id=part_type.id, sample_id=sample_id
                ),
            )
            all_parts += sample_parts
        self.browser.get(all_parts, "collections")

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
                ).append(Serializer.serialize(part, include="collections"))
        return data
