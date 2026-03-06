"""Tests for signal handling functionality."""

import json
import os
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Skip all tests on Windows - signals work differently there
pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Unix signal handling not available on Windows"
)

from gpudispatch.core.signals import SignalHandler, load_config_from_file


class TestLoadConfigFromFile:
    """Tests for config file loading."""

    def test_load_valid_config(self, tmp_path: Path):
        """Test loading a valid configuration file."""
        config_file = tmp_path / "gpu_config.json"
        config_data = {
            "available_gpus": [0, 1, 2, 3],
            "max_checks": 5,
            "memory_threshold_mb": 500
        }
        config_file.write_text(json.dumps(config_data))

        result = load_config_from_file(str(config_file))

        assert result == config_data
        assert result["available_gpus"] == [0, 1, 2, 3]
        assert result["max_checks"] == 5
        assert result["memory_threshold_mb"] == 500

    def test_load_nonexistent_file(self, tmp_path: Path):
        """Test loading from a non-existent file returns None."""
        config_file = tmp_path / "nonexistent.json"

        result = load_config_from_file(str(config_file))

        assert result is None

    def test_load_invalid_json(self, tmp_path: Path):
        """Test loading invalid JSON returns None."""
        config_file = tmp_path / "invalid.json"
        config_file.write_text("{ not valid json }")

        result = load_config_from_file(str(config_file))

        assert result is None

    def test_load_empty_file(self, tmp_path: Path):
        """Test loading empty file returns None."""
        config_file = tmp_path / "empty.json"
        config_file.write_text("")

        result = load_config_from_file(str(config_file))

        assert result is None

    def test_load_partial_config(self, tmp_path: Path):
        """Test loading partial config returns what's there."""
        config_file = tmp_path / "partial.json"
        config_data = {"available_gpus": [1, 2]}
        config_file.write_text(json.dumps(config_data))

        result = load_config_from_file(str(config_file))

        assert result == {"available_gpus": [1, 2]}


class TestSignalHandlerCreation:
    """Tests for SignalHandler initialization."""

    def test_create_signal_handler_no_dispatcher(self):
        """Test creating SignalHandler without dispatcher."""
        handler = SignalHandler()

        assert handler._dispatcher is None
        assert handler._config_path is None
        assert handler._handlers_installed is False

    def test_create_signal_handler_with_dispatcher(self):
        """Test creating SignalHandler with dispatcher."""
        mock_dispatcher = MagicMock()

        handler = SignalHandler(dispatcher=mock_dispatcher)

        assert handler._dispatcher is mock_dispatcher

    def test_create_signal_handler_with_config_path(self, tmp_path: Path):
        """Test creating SignalHandler with config path."""
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")

        handler = SignalHandler(config_path=str(config_file))

        assert handler._config_path == str(config_file)


class TestSignalHandlerInstall:
    """Tests for installing signal handlers."""

    def test_install_handlers(self):
        """Test installing signal handlers."""
        handler = SignalHandler()

        # Save original handlers
        original_sighup = signal.getsignal(signal.SIGHUP)
        original_sigusr1 = signal.getsignal(signal.SIGUSR1)
        original_sigterm = signal.getsignal(signal.SIGTERM)
        original_sigint = signal.getsignal(signal.SIGINT)

        try:
            handler.install()

            assert handler._handlers_installed is True
            # Verify handlers were changed
            assert signal.getsignal(signal.SIGHUP) != original_sighup
            assert signal.getsignal(signal.SIGUSR1) != original_sigusr1
            assert signal.getsignal(signal.SIGTERM) != original_sigterm
            assert signal.getsignal(signal.SIGINT) != original_sigint

        finally:
            # Restore original handlers
            handler.uninstall()
            signal.signal(signal.SIGHUP, original_sighup)
            signal.signal(signal.SIGUSR1, original_sigusr1)
            signal.signal(signal.SIGTERM, original_sigterm)
            signal.signal(signal.SIGINT, original_sigint)

    def test_install_twice_is_idempotent(self):
        """Test that installing handlers twice doesn't cause issues."""
        handler = SignalHandler()

        # Save original handlers
        original_sighup = signal.getsignal(signal.SIGHUP)

        try:
            handler.install()
            first_handler = signal.getsignal(signal.SIGHUP)

            handler.install()  # Install again
            second_handler = signal.getsignal(signal.SIGHUP)

            # Should be the same handler (idempotent)
            assert first_handler == second_handler

        finally:
            handler.uninstall()
            signal.signal(signal.SIGHUP, original_sighup)

    def test_uninstall_restores_defaults(self):
        """Test that uninstall restores default handlers."""
        handler = SignalHandler()

        # Save original handlers
        original_sigusr1 = signal.getsignal(signal.SIGUSR1)

        try:
            handler.install()
            handler.uninstall()

            # After uninstall, handlers_installed should be False
            assert handler._handlers_installed is False

        finally:
            signal.signal(signal.SIGUSR1, original_sigusr1)


class TestSIGHUPHandler:
    """Tests for SIGHUP (reload config) handling."""

    def test_sighup_triggers_reload_callback(self, tmp_path: Path):
        """Test that SIGHUP triggers the reload callback."""
        reload_called = threading.Event()

        def on_reload(config):
            reload_called.set()

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"available_gpus": [0, 1]}))

        handler = SignalHandler(
            config_path=str(config_file),
            on_reload=on_reload
        )

        try:
            handler.install()

            # Send SIGHUP to ourselves
            signal.raise_signal(signal.SIGHUP)

            # Give it a moment to process
            assert reload_called.wait(timeout=1.0), "Reload callback was not called"

        finally:
            handler.uninstall()

    def test_sighup_passes_config_to_callback(self, tmp_path: Path):
        """Test that SIGHUP passes loaded config to callback."""
        received_config = {}

        def on_reload(config):
            received_config.update(config or {})

        config_data = {"available_gpus": [2, 3], "memory_threshold_mb": 1000}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        handler = SignalHandler(
            config_path=str(config_file),
            on_reload=on_reload
        )

        try:
            handler.install()
            signal.raise_signal(signal.SIGHUP)

            time.sleep(0.1)  # Give it a moment

            assert received_config == config_data

        finally:
            handler.uninstall()

    def test_sighup_without_config_path(self):
        """Test SIGHUP when no config path is set."""
        reload_called = threading.Event()

        def on_reload(config):
            reload_called.set()

        handler = SignalHandler(on_reload=on_reload)

        try:
            handler.install()
            signal.raise_signal(signal.SIGHUP)

            # Should still call callback but with None
            assert reload_called.wait(timeout=1.0)

        finally:
            handler.uninstall()


class TestSIGUSR1Handler:
    """Tests for SIGUSR1 (drain mode) handling."""

    def test_sigusr1_triggers_drain_callback(self):
        """Test that SIGUSR1 triggers the drain callback."""
        drain_called = threading.Event()

        def on_drain():
            drain_called.set()

        handler = SignalHandler(on_drain=on_drain)

        try:
            handler.install()
            signal.raise_signal(signal.SIGUSR1)

            assert drain_called.wait(timeout=1.0), "Drain callback was not called"

        finally:
            handler.uninstall()

    def test_sigusr1_calls_dispatcher_drain(self):
        """Test that SIGUSR1 calls dispatcher.drain()."""
        mock_dispatcher = MagicMock()

        handler = SignalHandler(dispatcher=mock_dispatcher)

        try:
            handler.install()
            signal.raise_signal(signal.SIGUSR1)

            time.sleep(0.1)  # Give it a moment

            mock_dispatcher.drain.assert_called_once()

        finally:
            handler.uninstall()


class TestShutdownHandlers:
    """Tests for SIGTERM/SIGINT (shutdown) handling."""

    def test_sigterm_triggers_shutdown_callback(self):
        """Test that SIGTERM triggers the shutdown callback."""
        shutdown_called = threading.Event()

        def on_shutdown():
            shutdown_called.set()

        handler = SignalHandler(on_shutdown=on_shutdown)

        try:
            handler.install()
            signal.raise_signal(signal.SIGTERM)

            assert shutdown_called.wait(timeout=1.0), "Shutdown callback was not called"

        finally:
            handler.uninstall()

    def test_sigint_triggers_shutdown_callback(self):
        """Test that SIGINT triggers the shutdown callback."""
        shutdown_called = threading.Event()

        def on_shutdown():
            shutdown_called.set()

        handler = SignalHandler(on_shutdown=on_shutdown)

        try:
            handler.install()
            signal.raise_signal(signal.SIGINT)

            assert shutdown_called.wait(timeout=1.0), "Shutdown callback was not called"

        finally:
            handler.uninstall()

    def test_sigterm_calls_dispatcher_shutdown(self):
        """Test that SIGTERM calls dispatcher.shutdown()."""
        mock_dispatcher = MagicMock()

        handler = SignalHandler(dispatcher=mock_dispatcher)

        try:
            handler.install()
            signal.raise_signal(signal.SIGTERM)

            time.sleep(0.1)  # Give it a moment

            mock_dispatcher.shutdown.assert_called_once()

        finally:
            handler.uninstall()


class TestSignalHandlerWithDispatcher:
    """Integration tests with Dispatcher mock."""

    def test_full_signal_cycle(self, tmp_path: Path):
        """Test full cycle: create, install, handle signals, uninstall."""
        mock_dispatcher = MagicMock()

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"available_gpus": [0]}))

        handler = SignalHandler(
            dispatcher=mock_dispatcher,
            config_path=str(config_file)
        )

        try:
            handler.install()
            assert handler._handlers_installed is True

            # Test reload
            signal.raise_signal(signal.SIGHUP)
            time.sleep(0.1)

            # Test drain
            signal.raise_signal(signal.SIGUSR1)
            time.sleep(0.1)
            mock_dispatcher.drain.assert_called()

        finally:
            handler.uninstall()
            assert handler._handlers_installed is False


class TestThreadSafety:
    """Tests for thread-safe signal handling."""

    def test_handler_is_thread_safe(self):
        """Test that signal handling is thread-safe with locks."""
        call_count = 0
        lock = threading.Lock()

        def on_drain():
            nonlocal call_count
            with lock:
                call_count += 1

        handler = SignalHandler(on_drain=on_drain)

        try:
            handler.install()

            # Send multiple signals rapidly
            for _ in range(3):
                signal.raise_signal(signal.SIGUSR1)
                time.sleep(0.05)

            time.sleep(0.2)

            # All signals should have been handled
            with lock:
                assert call_count == 3

        finally:
            handler.uninstall()


class TestDispatcherReloadConfig:
    """Tests for Dispatcher.reload_config() method."""

    def test_reload_config_updates_gpus(self):
        """Test that reload_config updates GPU pool."""
        from gpudispatch.core.dispatcher import Dispatcher

        with patch('gpudispatch.core.dispatcher.detect_gpus') as mock:
            mock.return_value = []
            dispatcher = Dispatcher(gpus=[0, 1])

        assert dispatcher._gpu_indices == [0, 1]

        dispatcher.reload_config({"available_gpus": [2, 3, 4]})

        assert dispatcher._gpu_indices == [2, 3, 4]
        assert len(dispatcher.available_gpus) == 3

    def test_reload_config_updates_memory_threshold(self):
        """Test that reload_config updates memory threshold."""
        from gpudispatch.core.dispatcher import Dispatcher

        with patch('gpudispatch.core.dispatcher.detect_gpus') as mock:
            mock.return_value = []
            dispatcher = Dispatcher(gpus=[0], memory_threshold=500)

        assert dispatcher.memory_threshold_mb == 500

        dispatcher.reload_config({"memory_threshold_mb": 1000})

        assert dispatcher.memory_threshold_mb == 1000

    def test_reload_config_with_none(self):
        """Test that reload_config with None is a no-op."""
        from gpudispatch.core.dispatcher import Dispatcher

        with patch('gpudispatch.core.dispatcher.detect_gpus') as mock:
            mock.return_value = []
            dispatcher = Dispatcher(gpus=[0, 1])

        original_gpus = dispatcher._gpu_indices.copy()

        dispatcher.reload_config(None)

        assert dispatcher._gpu_indices == original_gpus


class TestDispatcherSetupSignals:
    """Tests for Dispatcher.setup_signals() method."""

    def test_setup_signals_returns_handler(self):
        """Test that setup_signals returns a SignalHandler."""
        from gpudispatch.core.dispatcher import Dispatcher

        with patch('gpudispatch.core.dispatcher.detect_gpus') as mock:
            mock.return_value = []
            dispatcher = Dispatcher(gpus=[0])

        try:
            handler = dispatcher.setup_signals()

            assert isinstance(handler, SignalHandler)
            assert handler._handlers_installed is True
            assert dispatcher._signal_handler is handler

        finally:
            handler.uninstall()

    def test_setup_signals_with_config_path(self, tmp_path: Path):
        """Test setup_signals with config path."""
        from gpudispatch.core.dispatcher import Dispatcher

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"available_gpus": [0, 1]}))

        with patch('gpudispatch.core.dispatcher.detect_gpus') as mock:
            mock.return_value = []
            dispatcher = Dispatcher(gpus=[0])

        try:
            handler = dispatcher.setup_signals(config_path=str(config_file))

            assert handler._config_path == str(config_file)

        finally:
            handler.uninstall()

    def test_setup_signals_connects_to_drain(self):
        """Test that SIGUSR1 calls dispatcher.drain() via setup_signals."""
        from gpudispatch.core.dispatcher import Dispatcher

        with patch('gpudispatch.core.dispatcher.detect_gpus') as mock:
            mock.return_value = []
            dispatcher = Dispatcher(gpus=[0])

        try:
            handler = dispatcher.setup_signals()

            assert not dispatcher._drain_event.is_set()

            signal.raise_signal(signal.SIGUSR1)
            time.sleep(0.1)

            assert dispatcher._drain_event.is_set()

        finally:
            handler.uninstall()

    def test_setup_signals_connects_to_reload(self, tmp_path: Path):
        """Test that SIGHUP calls dispatcher.reload_config() via setup_signals."""
        from gpudispatch.core.dispatcher import Dispatcher

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"available_gpus": [2, 3]}))

        with patch('gpudispatch.core.dispatcher.detect_gpus') as mock:
            mock.return_value = []
            dispatcher = Dispatcher(gpus=[0, 1])

        try:
            handler = dispatcher.setup_signals(config_path=str(config_file))

            assert dispatcher._gpu_indices == [0, 1]

            signal.raise_signal(signal.SIGHUP)
            time.sleep(0.1)

            assert dispatcher._gpu_indices == [2, 3]

        finally:
            handler.uninstall()
