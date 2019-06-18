from terrarium.adapters.aquarium.requester import DataRequester
from .builder_abc import BuilderABC


class SampleGraphBuilder(BuilderABC):
    def build(self, samples):
        return self.requester.build_sample_graph(samples)
