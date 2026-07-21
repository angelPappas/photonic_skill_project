from abc import ABC, abstractmethod


class PhotonicComponent(ABC):
    def __init__(self, **kwargs):
        self.name = kwargs["name"]
        self.model = kwargs["model"]
        self.database_path = kwargs["database_path"]
        self.interconnect_path = kwargs.get("interconnect_path")

    @abstractmethod
    def apply_changes():
        pass


class MMI(PhotonicComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def apply_changes():
        pass

class DirectionalCoupler(PhotonicComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def apply_changes():
        pass

class Waveguide(PhotonicComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def apply_changes():
        pass