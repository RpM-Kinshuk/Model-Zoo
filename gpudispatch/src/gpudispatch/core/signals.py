"""Signal handling for Dispatcher runtime control.

This module provides Unix signal handling to control the Dispatcher at runtime:
- SIGHUP: Reload configuration from file
- SIGUSR1: Enter drain mode (finish current jobs, don't accept new ones)
- SIGTERM/SIGINT: Graceful shutdown

Example usage:
    >>> from gpudispatch.core.signals import SignalHandler
    >>> from gpudispatch.core.dispatcher import Dispatcher
    >>>
    >>> dispatcher = Dispatcher(gpus=[0, 1])
    >>> handler = SignalHandler(
    ...     dispatcher=dispatcher,
    ...     config_path="gpu_config.json"
    ... )
    >>> handler.install()
    >>> # Now signals will control the dispatcher
    >>> # SIGHUP reloads config, SIGUSR1 drains, SIGTERM/SIGINT shutdown

Note:
    Signal handling only works on Unix systems. On Windows, this module
    provides no-op implementations.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from gpudispatch.core.dispatcher import Dispatcher

logger = logging.getLogger(__name__)

# Type alias for config dict
ConfigDict = Dict[str, Any]


def load_config_from_file(config_path: str) -> Optional[ConfigDict]:
    """Load configuration from a JSON file.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        Dictionary with configuration values, or None if file doesn't exist
        or cannot be parsed.

    Config file format:
        {
            "available_gpus": [0, 1, 2, 3],
            "max_checks": 5,
            "memory_threshold_mb": 500
        }
    """
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}")
        return None

    try:
        with open(config_path, "r") as f:
            content = f.read().strip()
            if not content:
                logger.warning(f"Config file is empty: {config_path}")
                return None
            return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file {config_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return None


class SignalHandler:
    """Handles Unix signals for Dispatcher runtime control.

    The SignalHandler registers handlers for the following signals:
    - SIGHUP: Reload configuration from the specified config file
    - SIGUSR1: Enter drain mode (finish current jobs, reject new ones)
    - SIGTERM: Graceful shutdown
    - SIGINT: Graceful shutdown (Ctrl+C)

    The handler can work with an optional Dispatcher instance, calling its
    drain() and shutdown() methods directly. Alternatively, custom callbacks
    can be provided for each signal type.

    Attributes:
        _dispatcher: Optional Dispatcher to control via signals.
        _config_path: Path to JSON config file for SIGHUP reload.
        _on_reload: Custom callback for config reload.
        _on_drain: Custom callback for drain mode.
        _on_shutdown: Custom callback for shutdown.
        _handlers_installed: Whether signal handlers are currently installed.

    Example:
        >>> handler = SignalHandler(
        ...     dispatcher=my_dispatcher,
        ...     config_path="gpu_config.json"
        ... )
        >>> handler.install()
        >>> # ... dispatcher runs ...
        >>> handler.uninstall()  # Restore original handlers
    """

    def __init__(
        self,
        dispatcher: Optional[Dispatcher] = None,
        config_path: Optional[str] = None,
        on_reload: Optional[Callable[[Optional[ConfigDict]], None]] = None,
        on_drain: Optional[Callable[[], None]] = None,
        on_shutdown: Optional[Callable[[], None]] = None,
    ):
        """Initialize the SignalHandler.

        Args:
            dispatcher: Optional Dispatcher instance to control via signals.
                If provided, drain() and shutdown() will be called on it.
            config_path: Path to JSON configuration file. On SIGHUP, this
                file will be loaded and passed to on_reload callback.
            on_reload: Callback function called on SIGHUP. Receives the
                loaded config dict (or None if load failed).
            on_drain: Callback function called on SIGUSR1.
            on_shutdown: Callback function called on SIGTERM/SIGINT.
        """
        self._dispatcher = dispatcher
        self._config_path = config_path
        self._on_reload = on_reload
        self._on_drain = on_drain
        self._on_shutdown = on_shutdown

        self._handlers_installed = False
        self._lock = threading.Lock()

        # Store original handlers for restoration
        self._original_handlers: Dict[int, Any] = {}

    def install(self) -> None:
        """Install signal handlers.

        This registers handlers for SIGHUP, SIGUSR1, SIGTERM, and SIGINT.
        On non-Unix systems (Windows), this is a no-op.

        The method is idempotent - calling it multiple times has no
        additional effect.
        """
        if sys.platform == "win32":
            logger.warning("Signal handling not supported on Windows")
            return

        with self._lock:
            if self._handlers_installed:
                return

            # Save original handlers
            self._original_handlers[signal.SIGTERM] = signal.getsignal(signal.SIGTERM)
            self._original_handlers[signal.SIGINT] = signal.getsignal(signal.SIGINT)

            if hasattr(signal, "SIGHUP"):
                self._original_handlers[signal.SIGHUP] = signal.getsignal(signal.SIGHUP)
                signal.signal(signal.SIGHUP, self._handle_sighup)

            if hasattr(signal, "SIGUSR1"):
                self._original_handlers[signal.SIGUSR1] = signal.getsignal(signal.SIGUSR1)
                signal.signal(signal.SIGUSR1, self._handle_sigusr1)

            signal.signal(signal.SIGTERM, self._handle_sigterm)
            signal.signal(signal.SIGINT, self._handle_sigint)

            self._handlers_installed = True
            logger.info("Signal handlers installed (SIGHUP=reload, SIGUSR1=drain, SIGTERM/SIGINT=shutdown)")

    def uninstall(self) -> None:
        """Uninstall signal handlers and restore originals.

        This restores the original signal handlers that were in place
        before install() was called.

        The method is idempotent - calling it multiple times has no
        additional effect.
        """
        if sys.platform == "win32":
            return

        with self._lock:
            if not self._handlers_installed:
                return

            # Restore original handlers
            for sig, handler in self._original_handlers.items():
                try:
                    signal.signal(sig, handler)
                except Exception as e:
                    logger.warning(f"Failed to restore handler for signal {sig}: {e}")

            self._original_handlers.clear()
            self._handlers_installed = False
            logger.info("Signal handlers uninstalled")

    def _handle_sighup(self, signum: int, frame: Any) -> None:
        """Handle SIGHUP signal - reload configuration.

        Args:
            signum: Signal number (should be SIGHUP).
            frame: Current stack frame.
        """
        logger.info(f"Received SIGHUP (signal {signum}), reloading configuration...")

        config = None
        if self._config_path:
            config = load_config_from_file(self._config_path)
            if config:
                logger.info(f"Configuration loaded: {config}")
            else:
                logger.warning("Failed to load configuration")

        if self._on_reload:
            try:
                self._on_reload(config)
            except Exception as e:
                logger.error(f"Error in reload callback: {e}")

    def _handle_sigusr1(self, signum: int, frame: Any) -> None:
        """Handle SIGUSR1 signal - enter drain mode.

        Args:
            signum: Signal number (should be SIGUSR1).
            frame: Current stack frame.
        """
        logger.info(f"Received SIGUSR1 (signal {signum}), entering drain mode...")

        if self._dispatcher:
            try:
                self._dispatcher.drain()
            except Exception as e:
                logger.error(f"Error calling dispatcher.drain(): {e}")

        if self._on_drain:
            try:
                self._on_drain()
            except Exception as e:
                logger.error(f"Error in drain callback: {e}")

    def _handle_sigterm(self, signum: int, frame: Any) -> None:
        """Handle SIGTERM signal - graceful shutdown.

        Args:
            signum: Signal number (should be SIGTERM).
            frame: Current stack frame.
        """
        logger.info(f"Received SIGTERM (signal {signum}), initiating shutdown...")
        self._do_shutdown()

    def _handle_sigint(self, signum: int, frame: Any) -> None:
        """Handle SIGINT signal - graceful shutdown (Ctrl+C).

        Args:
            signum: Signal number (should be SIGINT).
            frame: Current stack frame.
        """
        logger.info(f"Received SIGINT (signal {signum}), initiating shutdown...")
        self._do_shutdown()

    def _do_shutdown(self) -> None:
        """Perform shutdown actions."""
        if self._dispatcher:
            try:
                self._dispatcher.shutdown(wait=False)
            except Exception as e:
                logger.error(f"Error calling dispatcher.shutdown(): {e}")

        if self._on_shutdown:
            try:
                self._on_shutdown()
            except Exception as e:
                logger.error(f"Error in shutdown callback: {e}")
