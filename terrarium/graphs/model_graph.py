from .graphs import SchemaGraph
from terrarium.exceptions import SchemaValidationError
from terrarium.schemas.schemas import Schema, CustomSchemas


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
            )
        try:
            pk = data[Schema.PRIMARY_KEY]
        except KeyError as e:
            raise SchemaValidationError(
                "Primary key {} was not found in data {}".format(
                    Schema.PRIMARY_KEY, data
                )
            )
        return "{}_{}".format(model_class, pk)

    def shallow_copy(self):
        return self.__class__(name=self.name)

    def model_data(self, model_class):
        for n, ndata in self.node_data():
            if ndata[Schema.MODEL_CLASS] == model_class:
                yield n, ndata

    def models(self, model_class):
        for n, ndata in self.model_data(model_class):
            yield ndata


class SampleGraph(ModelGraph):

    SCHEMAS = [CustomSchemas.SAMPLE_SCHEMA.schema]


class AFTGraph(ModelGraph):

    SCHEMAS = [CustomSchemas.AFT_SCHEMA.schema]


class OperationGraph(ModelGraph):

    SCHEMAS = [CustomSchemas.AFT_SAMPLE_SCHEMA.schema, CustomSchemas.ITEM_SCHEMA.schema]
