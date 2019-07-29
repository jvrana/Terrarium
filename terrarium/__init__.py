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
    def __init__(self, adapter: AdapterABC):
        self.adapter = adapter

        # builders
        self.blueprint_builder = None
        self.operation_graph_builder = None

        # graphs
        self.blueprint_graph = None
        self.sample_graph = None
        self.operation_graph = None
