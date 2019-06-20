from abc import ABC, abstractmethod


class AdapterABC(ABC):
    """Adapter to a data server."""

    @abstractmethod
    def build_sample_graph(self):
        pass

    @abstractmethod
    def collect_afts_from_plans(self):
        pass

    @abstractmethod
    def collect_deployed_afts(self):
        pass

    @abstractmethod
    def collect_items(self):
        pass

    @abstractmethod
    def collect_parts(self):
        pass
