from terrarium.graphs import SampleGraph

from terrarium.utils.async_wrapper import make_async
from terrarium.schemas import Schema
from .adapterabc import AdapterABC
from uuid import uuid4

# typing
from pydent.sessionabc import SessionABC
from pydent.base import ModelBase
from pydent.models import FieldValue, AllowableFieldType, Plan
from networkx import DiGraph
from typing import List


class Serializer(object):
    """
    Serializes models.
    """

    @staticmethod
    def serialize(model: ModelBase, *args, **kwargs) -> dict:
        data = model.dump(*args, **kwargs)
        data[Schema.PRIMARY_KEY] = model._primary_key
        data[Schema.MODEL_CLASS] = model.__class__.__name__
        return data

    # TODO: operation tpe and field type serialization is unnecessary
    @classmethod
    def serialize_aft(cls, aft: AllowableFieldType) -> dict:
        if aft is None:
            return str(uuid4())
        data = cls.serialize(
            aft, include={"field_type": {"operation_type": "field_types"}}
        )
        return data

    @classmethod
    def serialize_iovalue(cls, fv: FieldValue) -> dict:
        data = cls.serialize(fv)
        data["allowable_field_type"] = cls.serialize_aft(fv.allowable_field_type)
        data["plan_id"] = fv.operation.plans[0].id
        return data


class AquariumAdapter(AdapterABC):
    """
    An adapter to the Aquarium fabrication server.
    """

    def __init__(self, session: SessionABC):
        assert session.using_requests is True
        assert session.using_cache is True
        assert session.browser
        self.session = session

    @property
    def browser(self):
        return self.session.browser

    @classmethod
    def build_sample_graph(
        cls, samples: List[ModelBase], g=None, visited=None
    ) -> DiGraph:
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

        browser = samples[0].session.browser
        browser.get(samples, {"field_values": "sample"})

        parent_samples = []
        for sample in samples:
            for fv in sample.field_values:
                if fv.sample:
                    parent_samples.append(fv.sample)
                    m1 = model_serializer(fv.sample)
                    m2 = model_serializer(sample)
                    g.add_edge_from_models(m1, m2)

        return cls.build_sample_graph(parent_samples, g=g, visited=visited)

    def collect_items(self, afts: List[dict], sample_ids: List[int]) -> List[dict]:
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

    def collect_io_values_from_plans(self, plans: List[Plan]) -> tuple:
        with self.session.with_cache(timeout=60) as sess:
            sess.browser.update_cache(plans)
            sess.browser.get("Plan", {"operations": {"field_values"}})
            sess.browser.get("Wire", {"source", "destination"})
            sess.browser.get("FieldValue", {"allowable_field_type": "field_type"})
            sess.browser.get("Operation", "operation_type")
            nodes = [
                Serializer.serialize_iovalue(fv)
                for fv in sess.browser.get(
                    "FieldValue", query={"parent_class": "Operation"}
                )
            ]
            edges = []
            for w in sess.browser.get("Wire"):
                if (
                    w.source
                    and w.source.allowable_field_type
                    and w.destination
                    and w.destination.allowable_field_type
                ):
                    src_io = Serializer.serialize_iovalue(w.source)
                    dest_io = Serializer.serialize_iovalue(w.destination)
                    edges.append((src_io, dest_io))
        return nodes, edges

    def collect_data_from_plans(self, plans: List[Plan]) -> tuple:
        return self.collect_io_values_from_plans(plans)

    def collect_deployed_afts(self) -> List[dict]:
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
