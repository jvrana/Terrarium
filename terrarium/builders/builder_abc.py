from abc import ABC, abstractmethod


class BuilderABC(ABC):
    def __init__(self, requester):
        self.requester = requester

    @abstractmethod
    def build(self):
        pass
