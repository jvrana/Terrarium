from abc import ABC, abstractmethod
from terrarium.adapters import AdapterABC


class BuilderABC(ABC):
    """
    Abstract base class for the graph builder class. Sets an Adapter instance, which
    provides a connection to a fabrication database.
    """

    def __init__(self, adapter: AdapterABC):
        self.adapter = adapter

    @abstractmethod
    def build(self):
        pass
