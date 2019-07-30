from abc import ABC, abstractmethod
from terrarium.adapters import AdapterABC
from terrarium.utils import logger


class BuilderABC(ABC):
    """
    Abstract base class for the graph builder class. Sets an Adapter instance, which
    provides a connection to a fabrication database.
    """

    def __init__(self, adapter: AdapterABC):
        self.adapter = adapter
        self._log = None

    @abstractmethod
    def build(self):
        pass

    @property
    def log(self):
        self._log = self._log or logger(self)
        return self._log
