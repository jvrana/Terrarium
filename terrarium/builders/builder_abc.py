from abc import ABC, abstractmethod
from terrarium.adapters import AdapterABC


class BuilderABC(ABC):
    def __init__(self, requester: AdapterABC):
        self.requester = requester

    @abstractmethod
    def build(self):
        pass
