"""Tests for resource abstractions."""

import pytest
from gpudispatch.core.resources import GPU, Memory, Resource, ResourceRequirements


class TestGPU:
    def test_gpu_creation_with_index(self):
        gpu = GPU(index=0)
        assert gpu.index == 0
        assert gpu.memory is None

    def test_gpu_creation_with_memory(self):
        gpu = GPU(index=1, memory="16GB")
        assert gpu.index == 1
        assert gpu.memory == 16 * 1024  # Stored in MB

    def test_gpu_memory_parsing_gb(self):
        gpu = GPU(index=0, memory="24GB")
        assert gpu.memory == 24 * 1024

    def test_gpu_memory_parsing_mb(self):
        gpu = GPU(index=0, memory="8192MB")
        assert gpu.memory == 8192

    def test_gpu_equality(self):
        gpu1 = GPU(index=0)
        gpu2 = GPU(index=0)
        gpu3 = GPU(index=1)
        assert gpu1 == gpu2
        assert gpu1 != gpu3

    def test_gpu_hash(self):
        gpu1 = GPU(index=0)
        gpu2 = GPU(index=0)
        assert hash(gpu1) == hash(gpu2)
        gpu_set = {gpu1, gpu2}
        assert len(gpu_set) == 1


class TestMemory:
    def test_memory_from_string_gb(self):
        mem = Memory.from_string("16GB")
        assert mem.mb == 16 * 1024

    def test_memory_from_string_mb(self):
        mem = Memory.from_string("4096MB")
        assert mem.mb == 4096

    def test_memory_from_int_mb(self):
        mem = Memory(mb=8192)
        assert mem.mb == 8192

    def test_memory_comparison(self):
        mem1 = Memory(mb=8192)
        mem2 = Memory(mb=16384)
        assert mem1 < mem2
        assert mem2 > mem1

    def test_memory_str(self):
        mem = Memory(mb=16384)
        assert str(mem) == "16.0GB"


class TestResourceRequirements:
    def test_requirements_simple(self):
        req = ResourceRequirements(gpu=1)
        assert req.gpu_count == 1
        assert req.memory is None

    def test_requirements_with_memory(self):
        req = ResourceRequirements(gpu=2, memory="32GB")
        assert req.gpu_count == 2
        assert req.memory.mb == 32 * 1024

    def test_requirements_validation_negative_gpu(self):
        with pytest.raises(ValueError, match="gpu_count must be non-negative"):
            ResourceRequirements(gpu=-1)

    def test_requirements_satisfies(self):
        req = ResourceRequirements(gpu=2, memory="16GB")
        available = [GPU(0, memory="24GB"), GPU(1, memory="24GB")]
        assert req.satisfies(available)

    def test_requirements_not_satisfies_count(self):
        req = ResourceRequirements(gpu=3)
        available = [GPU(0), GPU(1)]
        assert not req.satisfies(available)
