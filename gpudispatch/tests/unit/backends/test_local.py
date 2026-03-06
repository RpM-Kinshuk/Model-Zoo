"""Tests for LocalBackend implementation."""

import pytest
import threading
from unittest.mock import patch, MagicMock
from typing import List

from gpudispatch.backends.local import LocalBackend
from gpudispatch.backends.base import Backend
from gpudispatch.core.resources import GPU, Memory
from gpudispatch.utils.gpu import GPUInfo


class TestLocalBackendInstantiation:
    """Tests for LocalBackend creation with various configurations."""

    def test_inherits_from_backend(self):
        """LocalBackend inherits from Backend ABC."""
        assert issubclass(LocalBackend, Backend)

    def test_default_instantiation(self):
        """LocalBackend can be instantiated with defaults."""
        backend = LocalBackend()
        assert backend is not None

    def test_name_property(self):
        """LocalBackend has name 'local'."""
        backend = LocalBackend()
        assert backend.name == "local"

    def test_gpus_auto_detection(self):
        """gpus='auto' uses GPU detection."""
        with patch('gpudispatch.backends.local.detect_gpus') as mock_detect:
            mock_detect.return_value = [
                GPUInfo(0, "GPU0", 16000, 100, 0),
                GPUInfo(1, "GPU1", 16000, 100, 0),
            ]
            backend = LocalBackend(gpus="auto")
            backend.start()
            assert len(backend._gpu_pool) == 2
            backend.shutdown()

    def test_gpus_explicit_list(self):
        """gpus can be a list of indices."""
        backend = LocalBackend(gpus=[0, 2, 3])
        backend.start()
        assert set(gpu.index for gpu in backend._gpu_pool) == {0, 2, 3}
        backend.shutdown()

    def test_gpus_single_index(self):
        """gpus can be a single index."""
        backend = LocalBackend(gpus=[1])
        backend.start()
        assert len(backend._gpu_pool) == 1
        assert backend._gpu_pool[0].index == 1
        backend.shutdown()

    def test_memory_threshold_string(self):
        """memory_threshold accepts string like '500MB'."""
        backend = LocalBackend(memory_threshold="500MB")
        assert backend._memory_threshold_mb == 500

    def test_memory_threshold_gb_string(self):
        """memory_threshold accepts string like '2GB'."""
        backend = LocalBackend(memory_threshold="2GB")
        assert backend._memory_threshold_mb == 2048

    def test_memory_threshold_int(self):
        """memory_threshold accepts integer (MB)."""
        backend = LocalBackend(memory_threshold=1000)
        assert backend._memory_threshold_mb == 1000

    def test_polling_interval(self):
        """polling_interval is stored correctly."""
        backend = LocalBackend(polling_interval=10)
        assert backend._polling_interval == 10

    def test_process_mode_subprocess(self):
        """process_mode='subprocess' is accepted."""
        backend = LocalBackend(process_mode="subprocess")
        assert backend._process_mode == "subprocess"

    def test_process_mode_thread(self):
        """process_mode='thread' is accepted."""
        backend = LocalBackend(process_mode="thread")
        assert backend._process_mode == "thread"

    def test_invalid_process_mode_raises(self):
        """Invalid process_mode raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            LocalBackend(process_mode="invalid")
        assert "process_mode" in str(exc_info.value).lower()


class TestLocalBackendLifecycle:
    """Tests for start/shutdown lifecycle."""

    def test_is_running_false_initially(self):
        """Backend is not running before start."""
        backend = LocalBackend(gpus=[0])
        assert backend.is_running is False

    def test_start_makes_running(self):
        """Start makes backend running."""
        backend = LocalBackend(gpus=[0])
        backend.start()
        assert backend.is_running is True
        backend.shutdown()

    def test_shutdown_makes_not_running(self):
        """Shutdown makes backend not running."""
        backend = LocalBackend(gpus=[0])
        backend.start()
        backend.shutdown()
        assert backend.is_running is False

    def test_context_manager(self):
        """Backend works as context manager."""
        backend = LocalBackend(gpus=[0])
        with backend:
            assert backend.is_running is True
        assert backend.is_running is False

    def test_double_start_is_safe(self):
        """Starting twice does not error."""
        backend = LocalBackend(gpus=[0])
        backend.start()
        backend.start()  # Should not raise
        assert backend.is_running is True
        backend.shutdown()

    def test_shutdown_before_start_is_safe(self):
        """Shutdown before start does not error."""
        backend = LocalBackend(gpus=[0])
        backend.shutdown()  # Should not raise

    @patch('gpudispatch.backends.local.detect_gpus')
    def test_start_with_auto_gpus_detects(self, mock_detect):
        """Start with gpus='auto' calls detect_gpus."""
        mock_detect.return_value = [
            GPUInfo(0, "GPU0", 16000, 100, 0),
        ]
        backend = LocalBackend(gpus="auto")
        backend.start()
        mock_detect.assert_called()
        backend.shutdown()


class TestAllocateReleaseCycle:
    """Tests for allocate/release operations."""

    @patch('gpudispatch.backends.local.is_gpu_available', return_value=True)
    def test_allocate_single_gpu(self, mock_avail):
        """Can allocate a single GPU."""
        with LocalBackend(gpus=[0, 1]) as backend:
            gpus = backend.allocate_gpus(1)
            assert len(gpus) == 1
            assert isinstance(gpus[0], GPU)

    @patch('gpudispatch.backends.local.is_gpu_available', return_value=True)
    def test_allocate_multiple_gpus(self, mock_avail):
        """Can allocate multiple GPUs."""
        with LocalBackend(gpus=[0, 1, 2]) as backend:
            gpus = backend.allocate_gpus(2)
            assert len(gpus) == 2

    @patch('gpudispatch.backends.local.is_gpu_available', return_value=True)
    def test_allocate_returns_empty_when_insufficient(self, mock_avail):
        """Returns empty list when not enough GPUs."""
        with LocalBackend(gpus=[0]) as backend:
            gpus = backend.allocate_gpus(3)
            assert gpus == []

    @patch('gpudispatch.backends.local.is_gpu_available', return_value=True)
    def test_allocated_gpus_not_reallocated(self, mock_avail):
        """Allocated GPUs are tracked and not given out again."""
        with LocalBackend(gpus=[0, 1]) as backend:
            first = backend.allocate_gpus(1)
            second = backend.allocate_gpus(1)
            assert len(first) == 1
            assert len(second) == 1
            assert first[0].index != second[0].index

    @patch('gpudispatch.backends.local.is_gpu_available', return_value=True)
    def test_release_makes_gpu_available_again(self, mock_avail):
        """Released GPUs become available."""
        with LocalBackend(gpus=[0]) as backend:
            gpus = backend.allocate_gpus(1)
            assert len(backend.list_available()) == 0

            backend.release_gpus(gpus)
            assert len(backend.list_available()) == 1

    @patch('gpudispatch.backends.local.is_gpu_available', return_value=True)
    def test_release_allows_reallocation(self, mock_avail):
        """Released GPUs can be reallocated."""
        with LocalBackend(gpus=[0]) as backend:
            gpus1 = backend.allocate_gpus(1)
            backend.release_gpus(gpus1)
            gpus2 = backend.allocate_gpus(1)
            assert len(gpus2) == 1

    @patch('gpudispatch.backends.local.is_gpu_available', return_value=True)
    def test_release_unallocated_is_noop(self, mock_avail):
        """Releasing unallocated GPU does not error."""
        with LocalBackend(gpus=[0]) as backend:
            backend.release_gpus([GPU(99)])  # Should not raise

    def test_allocate_with_memory_checks_threshold(self):
        """Allocation respects memory threshold for availability."""
        with patch('gpudispatch.backends.local.is_gpu_available') as mock_avail:
            # GPU 0 has too much memory in use, GPU 1 is free
            mock_avail.side_effect = lambda idx, threshold_mb: idx == 1

            with LocalBackend(gpus=[0, 1], memory_threshold="500MB") as backend:
                gpus = backend.allocate_gpus(1)
                assert len(gpus) == 1
                assert gpus[0].index == 1

    @patch('gpudispatch.backends.local.is_gpu_available', return_value=True)
    def test_shutdown_releases_all_gpus(self, mock_avail):
        """Shutdown releases any allocated GPUs."""
        backend = LocalBackend(gpus=[0, 1])
        backend.start()
        backend.allocate_gpus(2)
        backend.shutdown()
        # After shutdown, occupied set should be cleared
        assert len(backend._occupied_gpus) == 0


class TestListAvailable:
    """Tests for list_available method."""

    @patch('gpudispatch.backends.local.is_gpu_available', return_value=True)
    def test_list_available_returns_all_initially(self, mock_avail):
        """All GPUs available before any allocation."""
        with LocalBackend(gpus=[0, 1, 2]) as backend:
            available = backend.list_available()
            assert len(available) == 3

    @patch('gpudispatch.backends.local.is_gpu_available', return_value=True)
    def test_list_available_excludes_occupied(self, mock_avail):
        """Occupied GPUs excluded from available."""
        with LocalBackend(gpus=[0, 1]) as backend:
            backend.allocate_gpus(1)
            available = backend.list_available()
            assert len(available) == 1

    def test_list_available_checks_memory_threshold(self):
        """list_available checks memory threshold."""
        with patch('gpudispatch.backends.local.is_gpu_available') as mock_avail:
            # Only GPU 1 is under threshold
            mock_avail.side_effect = lambda idx, threshold_mb: idx == 1

            with LocalBackend(gpus=[0, 1], memory_threshold="500MB") as backend:
                available = backend.list_available()
                # Only GPU 1 should be listed as available
                assert len(available) == 1
                assert available[0].index == 1

    @patch('gpudispatch.backends.local.is_gpu_available', return_value=True)
    def test_list_available_returns_gpu_objects(self, mock_avail):
        """list_available returns GPU objects."""
        with LocalBackend(gpus=[0]) as backend:
            available = backend.list_available()
            assert all(isinstance(g, GPU) for g in available)


class TestHealthCheck:
    """Tests for health_check method."""

    def test_health_check_false_before_start(self):
        """Health check returns False before start."""
        backend = LocalBackend(gpus=[0])
        assert backend.health_check() is False

    def test_health_check_true_when_running(self):
        """Health check returns True when running."""
        with LocalBackend(gpus=[0]) as backend:
            assert backend.health_check() is True

    def test_health_check_false_after_shutdown(self):
        """Health check returns False after shutdown."""
        backend = LocalBackend(gpus=[0])
        backend.start()
        backend.shutdown()
        assert backend.health_check() is False


class TestConcurrentAccess:
    """Tests for thread safety."""

    @patch('gpudispatch.backends.local.is_gpu_available', return_value=True)
    def test_concurrent_allocations(self, mock_avail):
        """Multiple threads can allocate without corruption."""
        backend = LocalBackend(gpus=[0, 1, 2, 3])
        backend.start()

        allocated: List[GPU] = []
        lock = threading.Lock()
        errors: List[Exception] = []

        def allocate_one():
            try:
                gpus = backend.allocate_gpus(1)
                with lock:
                    allocated.extend(gpus)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=allocate_one) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        backend.shutdown()

        assert len(errors) == 0
        assert len(allocated) == 4
        # Each GPU should be unique
        indices = [g.index for g in allocated]
        assert len(set(indices)) == 4

    @patch('gpudispatch.backends.local.is_gpu_available', return_value=True)
    def test_concurrent_allocate_release(self, mock_avail):
        """Concurrent allocate and release operations are safe."""
        backend = LocalBackend(gpus=[0, 1])
        backend.start()

        errors: List[Exception] = []

        def allocate_and_release():
            try:
                for _ in range(10):
                    gpus = backend.allocate_gpus(1)
                    if gpus:
                        backend.release_gpus(gpus)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=allocate_and_release) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        backend.shutdown()

        assert len(errors) == 0

    @patch('gpudispatch.backends.local.is_gpu_available', return_value=True)
    def test_list_available_thread_safe(self, mock_avail):
        """list_available is thread-safe during concurrent operations."""
        backend = LocalBackend(gpus=[0, 1, 2, 3])
        backend.start()

        errors: List[Exception] = []

        def list_and_allocate():
            try:
                for _ in range(10):
                    backend.list_available()
                    gpus = backend.allocate_gpus(1)
                    if gpus:
                        backend.release_gpus(gpus)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=list_and_allocate) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        backend.shutdown()

        assert len(errors) == 0


class TestAllocateWithMemoryRequirement:
    """Tests for allocate_gpus with memory parameter."""

    @patch('gpudispatch.backends.local.is_gpu_available', return_value=True)
    def test_allocate_with_memory_requirement(self, mock_avail):
        """Can allocate GPUs with memory requirement."""
        with LocalBackend(gpus=[0, 1]) as backend:
            # This tests the memory parameter passed to allocate_gpus
            # (not the threshold for considering a GPU "free")
            memory = Memory.from_string("8GB")
            gpus = backend.allocate_gpus(1, memory=memory)
            # Should return GPUs (memory requirement is stored, not validated against real GPU memory)
            assert len(gpus) == 1

    @patch('gpudispatch.backends.local.is_gpu_available', return_value=True)
    def test_allocate_with_none_memory(self, mock_avail):
        """allocate_gpus works with memory=None."""
        with LocalBackend(gpus=[0]) as backend:
            gpus = backend.allocate_gpus(1, memory=None)
            assert len(gpus) == 1


class TestProcessMode:
    """Tests for process_mode configuration."""

    def test_subprocess_mode_stored(self):
        """subprocess mode is stored correctly."""
        backend = LocalBackend(process_mode="subprocess")
        assert backend._process_mode == "subprocess"

    def test_thread_mode_stored(self):
        """thread mode is stored correctly."""
        backend = LocalBackend(process_mode="thread")
        assert backend._process_mode == "thread"

    def test_default_process_mode(self):
        """Default process_mode is subprocess."""
        backend = LocalBackend()
        assert backend._process_mode == "subprocess"
