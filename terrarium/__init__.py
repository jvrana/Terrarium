from terrarium.builders import OperationBlueprintBuilder, OperationGraphBuilder
from terrarium.adapters import AdapterABC
from terrarium import constants
from terrarium.__version__ import (
    __version__,
    __title__,
    __authors__,
    __homepage__,
    __repo__,
)


class Terrarium(object):

    __version__ = {"Terrarium": __version__}

    def __init__(self, adapter: AdapterABC):
        self.adapter = adapter

        # builders
        self.blueprint_builder = None
        self.operation_graph_builder = None

        # graphs
        self.blueprint_graph = None
        self.sample_graph = None
        self.operation_graph = None

    def info(self):
        return {"version": self.__version__, "adapter": self.adapter.info()}
