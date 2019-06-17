from .requester import DataRequester
from .builder_abc import BuilderABC


class SampleGraphBuilder(BuilderABC):
    def build(self, samples):
        return DataRequester.build_sample_graph(samples)
