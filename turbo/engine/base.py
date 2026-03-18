from abc import ABC, abstractmethod


class Engine(ABC):
    @abstractmethod
    def __init__(self, **kwargs) -> None: ...
