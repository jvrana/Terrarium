from abc import ABC, abstractmethod


class AdapterABC(ABC):
    """Adapter to a data server (e.g. a fabrication database)."""

    @abstractmethod
    def api_version(self):
        """Return the server identifier (including version of API)"""
        pass

    @abstractmethod
    def api_url(self):
        pass

    @abstractmethod
    def api_name(self):
        pass

    def info(self):
        return {
            "name": str(self.__class__),
            "api_version": self.api_version(),
            "api_url": self.api_url(),
            "api_name": self.api_name(),
        }

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

    def __str__(self):
        return "<{cls} api={name}(v{ver})@{url}>".format(
            cls=self.__class__.__name__,
            name=self.api_name(),
            url=self.api_url(),
            ver=self.api_version(),
        )

    def __repr__(self):
        return str(self)
