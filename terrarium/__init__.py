from terrarium.builders import OperationBlueprintBuilder, OperationGraphBuilder
from terrarium.graphs import AFTGraph, SampleGraph
from terrarium.adapters import AdapterABC
from terrarium import constants
from loggable import Loggable
from terrarium.__version__ import (
    __version__,
    __title__,
    __authors__,
    __homepage__,
    __repo__,
)


class TerrariumError(Exception):
    """Generic exception"""


class Terrarium(object):
    def __init__(
        self, adapter: AdapterABC, bp_graph: AFTGraph, sample_graph: SampleGraph
    ):
        self.log = Loggable(self)
        self.adapter = adapter
        self.blueprint = bp_graph
        self.sample_graph = sample_graph
        self.builder = OperationGraphBuilder(
            self.adapter, self.blueprint, self.sample_graph
        )
        self.graph = None

    def build(self):
        self.graph = self.builder.build()
        return self.graph


class TerrariumBlueprint(object):

    __version__ = {"Terrarium": __version__}

    def __init__(self, adapter: AdapterABC):
        self.log = Loggable(self)
        self.adapter = adapter

        # builders
        self.blueprint_builder = OperationBlueprintBuilder(adapter)

        # graphs
        self.blueprint_graph = None

    def train(self, plans):
        self.log.info("Training...")
        self.blueprint_builder.collect_deployed()
        self.blueprint_builder.collect(plans)
        self.blueprint_graph = self.blueprint_builder.build()

    # TODO: better API for this
    def new_goal(self, samples):
        sample_graph = self.adapter.build_sample_graph(samples)
        if self.blueprint_graph is None:
            raise TerrariumError("Terrarium has no training data. Please run 'train'.")
        return Terrarium(
            adapter=self.adapter,
            bp_graph=self.blueprint_graph,
            sample_graph=sample_graph,
        )

    def info(self):
        return {"version": self.__version__, "adapter": self.adapter.info()}
