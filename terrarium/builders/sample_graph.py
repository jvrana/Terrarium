from terrarium.requester import DataRequester


class SampleGraphBuilder(object):
    @staticmethod
    def build(samples):
        return DataRequester.build_sample_graph(samples)
