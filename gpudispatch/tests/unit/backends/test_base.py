"""Tests for Backend abstract base class."""

import pytest
from typing import Optional, List

from gpudispatch.backends.base import Backend
from gpudispatch.core.resources import GPU, Memory


class TestBackendCannotBeInstantiated:
    """Verify that Backend ABC cannot be instantiated directly."""

    def test_backend_is_abstract(self):
        """Backend cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            Backend()
        assert "abstract" in str(exc_info.value).lower()

    def test_backend_requires_all_abstract_methods(self):
        """Partial implementation still cannot be instantiated."""

        class PartialBackend(Backend):
            @property
            def name(self) -> str:
                return "partial"

        with pytest.raises(TypeError) as exc_info:
            PartialBackend()
        assert "abstract" in str(exc_info.value).lower()


class MockBackend(Backend):
    """A concrete implementation for testing the interface."""

    def __init__(self, gpus: Optional[List[GPU]] = None):
        self._gpus = gpus or [GPU(0), GPU(1)]
        self._allocated: List[GPU] = []
        self._running = False
        self._started = False
        self._shutdown_called = False

    @property
    def name(self) -> str:
        return "mock"

    @property
    def is_running(self) -> bool:
        return self._running

    def allocate_gpus(
        self, count: int, memory: Optional[Memory] = None
    ) -> List[GPU]:
        available = [g for g in self._gpus if g not in self._allocated]

        if memory is not None:
            available = [
                g for g in available
                if g.memory is not None and g.memory >= memory.mb
            ]

        if len(available) < count:
            return []

        allocated = available[:count]
        self._allocated.extend(allocated)
        return allocated

    def release_gpus(self, gpus: List[GPU]) -> None:
        for gpu in gpus:
            if gpu in self._allocated:
                self._allocated.remove(gpu)

    def list_available(self) -> List[GPU]:
        return [g for g in self._gpus if g not in self._allocated]

    def health_check(self) -> bool:
        return self._started and not self._shutdown_called

    def start(self) -> None:
        self._started = True
        self._running = True

    def shutdown(self) -> None:
        self._shutdown_called = True
        self._running = False


class TestMockBackendInterface:
    """Verify the interface contract through a mock implementation."""

    def test_mock_backend_instantiation(self):
        """Mock backend can be instantiated."""
        backend = MockBackend()
        assert backend is not None

    def test_name_property(self):
        """Backend has a name property."""
        backend = MockBackend()
        assert backend.name == "mock"
        assert isinstance(backend.name, str)

    def test_is_running_property(self):
        """Backend tracks running state."""
        backend = MockBackend()
        assert backend.is_running is False

        backend.start()
        assert backend.is_running is True

        backend.shutdown()
        assert backend.is_running is False


class TestAllocateGpus:
    """Tests for GPU allocation."""

    def test_allocate_single_gpu(self):
        """Can allocate a single GPU."""
        backend = MockBackend()
        gpus = backend.allocate_gpus(1)
        assert len(gpus) == 1
        assert isinstance(gpus[0], GPU)

    def test_allocate_multiple_gpus(self):
        """Can allocate multiple GPUs."""
        backend = MockBackend(gpus=[GPU(0), GPU(1), GPU(2)])
        gpus = backend.allocate_gpus(2)
        assert len(gpus) == 2

    def test_allocate_with_memory_requirement(self):
        """Can allocate GPUs with memory requirement."""
        backend = MockBackend(gpus=[GPU(0, memory="16GB"), GPU(1, memory="24GB")])
        gpus = backend.allocate_gpus(1, memory=Memory.from_string("20GB"))
        assert len(gpus) == 1
        assert gpus[0].index == 1  # Only GPU 1 has 24GB

    def test_allocate_returns_empty_when_insufficient(self):
        """Returns empty list when not enough GPUs available."""
        backend = MockBackend(gpus=[GPU(0)])
        gpus = backend.allocate_gpus(3)
        assert gpus == []

    def test_allocate_tracks_allocated_gpus(self):
        """Allocated GPUs are tracked and not reallocated."""
        backend = MockBackend(gpus=[GPU(0), GPU(1)])
        first = backend.allocate_gpus(1)
        second = backend.allocate_gpus(1)
        assert len(first) == 1
        assert len(second) == 1
        assert first[0] != second[0]

    def test_allocate_exhausts_pool(self):
        """Cannot allocate when pool is exhausted."""
        backend = MockBackend(gpus=[GPU(0), GPU(1)])
        backend.allocate_gpus(2)
        gpus = backend.allocate_gpus(1)
        assert gpus == []


class TestReleaseGpus:
    """Tests for GPU release."""

    def test_release_makes_gpus_available(self):
        """Released GPUs become available again."""
        backend = MockBackend(gpus=[GPU(0)])
        gpus = backend.allocate_gpus(1)
        assert len(backend.list_available()) == 0

        backend.release_gpus(gpus)
        assert len(backend.list_available()) == 1

    def test_release_allows_reallocation(self):
        """Released GPUs can be reallocated."""
        backend = MockBackend(gpus=[GPU(0)])
        gpus1 = backend.allocate_gpus(1)
        backend.release_gpus(gpus1)
        gpus2 = backend.allocate_gpus(1)
        assert len(gpus2) == 1

    def test_release_unallocated_gpu_is_noop(self):
        """Releasing unallocated GPU does not error."""
        backend = MockBackend(gpus=[GPU(0)])
        # Should not raise
        backend.release_gpus([GPU(99)])


class TestListAvailable:
    """Tests for listing available GPUs."""

    def test_list_available_returns_all_initially(self):
        """All GPUs are available initially."""
        backend = MockBackend(gpus=[GPU(0), GPU(1), GPU(2)])
        available = backend.list_available()
        assert len(available) == 3

    def test_list_available_excludes_allocated(self):
        """Allocated GPUs are excluded from available list."""
        backend = MockBackend(gpus=[GPU(0), GPU(1)])
        backend.allocate_gpus(1)
        available = backend.list_available()
        assert len(available) == 1

    def test_list_available_returns_gpu_objects(self):
        """Available list contains GPU objects."""
        backend = MockBackend(gpus=[GPU(0)])
        available = backend.list_available()
        assert all(isinstance(g, GPU) for g in available)


class TestHealthCheck:
    """Tests for health check."""

    def test_health_check_false_before_start(self):
        """Health check returns False before start."""
        backend = MockBackend()
        assert backend.health_check() is False

    def test_health_check_true_when_running(self):
        """Health check returns True when running."""
        backend = MockBackend()
        backend.start()
        assert backend.health_check() is True

    def test_health_check_false_after_shutdown(self):
        """Health check returns False after shutdown."""
        backend = MockBackend()
        backend.start()
        backend.shutdown()
        assert backend.health_check() is False


class TestBackendLifecycle:
    """Tests for start and shutdown lifecycle."""

    def test_start_activates_backend(self):
        """Start makes the backend running."""
        backend = MockBackend()
        assert backend.is_running is False
        backend.start()
        assert backend.is_running is True

    def test_shutdown_deactivates_backend(self):
        """Shutdown makes the backend not running."""
        backend = MockBackend()
        backend.start()
        backend.shutdown()
        assert backend.is_running is False

    def test_context_manager_protocol(self):
        """Backend supports context manager protocol."""
        backend = MockBackend()

        with backend:
            assert backend.is_running is True

        assert backend.is_running is False


class TestBackendTypeAnnotations:
    """Verify type annotations are correct."""

    def test_allocate_gpus_accepts_optional_memory(self):
        """allocate_gpus accepts None for memory."""
        backend = MockBackend()
        gpus = backend.allocate_gpus(1, memory=None)
        assert len(gpus) == 1

    def test_allocate_gpus_accepts_memory_object(self):
        """allocate_gpus accepts Memory object."""
        backend = MockBackend(gpus=[GPU(0, memory="32GB")])
        memory = Memory.from_string("16GB")
        gpus = backend.allocate_gpus(1, memory=memory)
        assert len(gpus) == 1
