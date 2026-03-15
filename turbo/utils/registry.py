from __future__ import annotations

from typing import Callable, Generic, Iterable, List, TypeVar

T = TypeVar("T")

class Registry(Generic[T]):
    def __init__(self, type: str):
        self._registry = {}
        self._type = type

    def register(self, name: str) -> Callable[[T], None]:
        if name in self._registry:
            raise ValueError(f"Name {name} already registered")

        def decorator(item: T) -> None:
            self._registry[name] = item

        return decorator

    def __getitem__(self, name: str) -> T:
        if name not in self._registry:
            raise ValueError(f"Name {name} not registered")
        return self._registry[name]

    def support_names(self) -> List[str]:
        return list(self._registry.keys())

    def assert_supported_name(self, names: str | Iterable[str]) -> None:
        if isinstance(names, str):
            names = [names]
        for n in names:
            if n not in self._registry:
                from argparse import ArgumentTypeError
                raise ArgumentTypeError(
                    f"Unsupported {self._type}: {n}. "
                    f"Supported items: {self.supported_names()}"
                )
