from abc import ABC, abstractmethod


class AdapterABC(ABC):
    """Adapter to a data server (e.g. a fabrication database)."""

    @abstractmethod
    def build_sample_graph(self):
        pass

    @abstractmethod
    def collect_data_from_plans(self):
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
