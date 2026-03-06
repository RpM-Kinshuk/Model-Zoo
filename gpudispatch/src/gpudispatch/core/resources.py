"""Resource abstractions for GPU and memory management."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union


@dataclass(frozen=True)
class Memory:
    """Memory specification in megabytes."""

    mb: int

    @classmethod
    def from_string(cls, value: str) -> Memory:
        """Parse memory string like '16GB' or '4096MB'."""
        match = re.match(r"^(\d+(?:\.\d+)?)\s*(GB|MB|gb|mb)$", value.strip())
        if not match:
            raise ValueError(f"Invalid memory format: {value}. Use '16GB' or '4096MB'.")

        amount = float(match.group(1))
        unit = match.group(2).upper()

        if unit == "GB":
            return cls(mb=int(amount * 1024))
        return cls(mb=int(amount))

    def __str__(self) -> str:
        if self.mb >= 1024:
            return f"{self.mb / 1024:.1f}GB"
        return f"{self.mb}MB"

    def __lt__(self, other: Memory) -> bool:
        return self.mb < other.mb

    def __le__(self, other: Memory) -> bool:
        return self.mb <= other.mb

    def __gt__(self, other: Memory) -> bool:
        return self.mb > other.mb

    def __ge__(self, other: Memory) -> bool:
        return self.mb >= other.mb


class Resource:
    """Base class for compute resources."""
    pass


@dataclass(frozen=True)
class GPU(Resource):
    """GPU resource specification."""

    index: int
    memory: Optional[int] = None  # Memory in MB

    def __init__(self, index: int, memory: Optional[Union[str, int]] = None):
        object.__setattr__(self, 'index', index)

        if memory is None:
            object.__setattr__(self, 'memory', None)
        elif isinstance(memory, str):
            parsed = Memory.from_string(memory)
            object.__setattr__(self, 'memory', parsed.mb)
        else:
            object.__setattr__(self, 'memory', memory)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GPU):
            return NotImplemented
        return self.index == other.index

    def __hash__(self) -> int:
        return hash(self.index)

    def __str__(self) -> str:
        if self.memory:
            return f"GPU({self.index}, {Memory(self.memory)})"
        return f"GPU({self.index})"


@dataclass
class ResourceRequirements:
    """Resource requirements for a job."""

    gpu_count: int = 0
    memory: Optional[Memory] = None

    def __init__(
        self,
        gpu: int = 0,
        memory: Optional[Union[str, Memory]] = None,
    ):
        if gpu < 0:
            raise ValueError("gpu_count must be non-negative")

        self.gpu_count = gpu

        if memory is None:
            self.memory = None
        elif isinstance(memory, str):
            self.memory = Memory.from_string(memory)
        else:
            self.memory = memory

    def satisfies(self, available: Sequence[GPU]) -> bool:
        """Check if available resources satisfy these requirements."""
        if len(available) < self.gpu_count:
            return False

        if self.memory is not None:
            for gpu in available[:self.gpu_count]:
                if gpu.memory is not None and gpu.memory < self.memory.mb:
                    return False

        return True
