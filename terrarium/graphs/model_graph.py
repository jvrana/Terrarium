from .graphs import SchemaGraph
from terrarium.exceptions import SchemaValidationError
from terrarium.schemas.schemas import Schema, CustomSchemas

# TODO: make a new model_graph that takes in a base_graph
# TODO: schemas should be handled by the serialization/deserialization library


class ModelGraph(SchemaGraph):

    SCHEMAS = []

    def __init__(self, name=None, graph=None):
        super().__init__(self._get_model_id, name=name, graph=graph)
        self.schemas = [s.copy() for s in self.SCHEMAS]

    @staticmethod
    def _get_model_id(data):
        try:
            model_class = data[Schema.MODEL_CLASS]
        except KeyError as e:
            raise SchemaValidationError(
                "Class key {} was not found in data {}".format(Schema.MODEL_CLASS, data)
            ) from e
        try:
            pk = data[Schema.PRIMARY_KEY]
        except KeyError as e:
            raise SchemaValidationError(
                "Primary key {} was not found in data {}".format(
                    Schema.PRIMARY_KEY, data
                )
            ) from e
        return "{}_{}".format(model_class, pk)

    def shallow_copy(self):
        return self.__class__(name=self.name)

    def model_data(self, model_class=None, filters=None):
        if filters is None:
            filters = []
        elif callable(filters):
            filters = [filters]
        if model_class:
            all_filters = [lambda x: x[Schema.MODEL_CLASS] == model_class] + filters
        else:
            all_filters = filters[:]
        return self.data_filter(filters=all_filters)

    def models(self, model_class):
        for n, ndata in self.model_data(model_class):
            yield ndata


class SampleGraph(ModelGraph):

    SCHEMAS = [CustomSchemas.sample_schema]


class AFTGraph(ModelGraph):

    SCHEMAS = [CustomSchemas.aft_schema]


class OperationGraph(ModelGraph):

    SCHEMAS = [CustomSchemas.aft_sample_schema, CustomSchemas.item_schema]
