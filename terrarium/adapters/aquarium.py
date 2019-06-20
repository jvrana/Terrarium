from terrarium.graphs import SampleGraph
from typing import Sequence
from terrarium.utils.async_wrapper import make_async
from terrarium.schemas import Schema
from .adapterabc import AdapterABC


class Serializer(object):
    """
    Serializes models.
    """

    @staticmethod
    def serialize(model, *args, **kwargs):
        data = model.dump(*args, **kwargs)
        data[Schema.PRIMARY_KEY] = model._primary_key
        data[Schema.MODEL_CLASS] = model.__class__.__name__
        return data

    # TODO: operation tpe and field type serialization is unnecessary
    @classmethod
    def serialize_aft(cls, aft):
        data = cls.serialize(
            aft, include={"field_type": {"operation_type": "field_types"}}
        )
        return data


class AquariumAdapter(AdapterABC):
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
        items = [i for i in items if i.location != "deleted"]
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
        part_type = self.session.ObjectType.find_by_name("__Part")

        @make_async(1, progress_bar=False)
        def parts_from_sample_ids(sample_ids):
            parts_list = []
            for sample_id in sample_ids:
                parts_list += self.session.Item.last(
                    lim, query={"object_type_id": part_type.id, "sample_id": sample_id}
                )
            return parts_list

        parts_list = parts_from_sample_ids(sample_ids)
        self.session.browser.get(parts_list, "collections")
        parts_list = [
            part
            for part in parts_list
            if part.collections and part.collections[0].location != "deleted"
        ]

        return [
            Serializer.serialize(part, include="collections") for part in parts_list
        ]

    def collect_afts_from_plans(self, plans):
        with self.session.with_cache(timeout=60) as sess:
            sess.browser.update_cache(plans)
            sess.browser.get("Plan", {"operations": {"field_values"}})
            sess.browser.get("Wire", {"source", "destination"})
            sess.browser.get("FieldValue", {"allowable_field_type": "field_type"})
            sess.browser.get("Operation", "operation_type")
            nodes = [
                Serializer.serialize_aft(aft)
                for aft in sess.browser.get("AllowableFieldType")
            ]
            edges = []
            for w in sess.browser.get("Wire"):
                src_aft = Serializer.serialize_aft(w.source.allowable_field_type)
                dest_aft = Serializer.serialize_aft(w.destination.allowable_field_type)
                edges.append((src_aft, dest_aft))
        return nodes, edges

    def collect_deployed_afts(self):
        with self.session.with_cache(timeout=60) as sess:
            sess.OperationType.where({"deployed": True})

            sess.browser.get(
                "OperationType",
                {
                    "field_types": {
                        "allowable_field_types": {"field_type": []},
                        "operation_type": "field_types",
                    }
                },
            )

            afts = sess.browser.get("AllowableFieldType")

        return [Serializer.serialize_aft(aft) for aft in afts]
