from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Capability:
    name: str
    description: str
    run: Callable
    available: Callable[[], bool]
    external: bool
    produces_observation: bool
    requires_memory: int
    requires_language: bool
    warmup_priority: float
    policy_train: bool = True


class CapabilityRegistry:
    def __init__(self):
        self._items: list[Capability] = []

    def register(self, item: Capability) -> None:
        if any(x.name == item.name for x in self._items):
            raise ValueError(f"duplicate capability: {item.name}")
        self._items.append(item)

    def names(self) -> list[str]:
        return [x.name for x in self._items]

    def get(self, index: int) -> Capability:
        return self._items[index]

    def index(self, name: str) -> int:
        return self.names().index(name)

    def items(self) -> list[Capability]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)
