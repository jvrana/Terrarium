from abc import ABC, abstractmethod
from terrarium.adapters import AdapterABC


class BuilderABC(ABC):
    def __init__(self, adapter: AdapterABC):
        self.adapter = adapter

    @abstractmethod
    def build(self):
        pass
