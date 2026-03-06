"""Tests for observability hooks."""

import logging
import pytest
from gpudispatch.observability.hooks import EventHook, HookRegistry, LoggingHook, hooks


class TestEventHook:
    def test_default_hook_has_no_callbacks(self):
        hook = EventHook()
        assert hook.on_job_start is None
        assert hook.on_job_complete is None
        assert hook.on_job_failed is None
        assert hook.on_experiment_start is None
        assert hook.on_experiment_complete is None

    def test_hook_with_callbacks(self):
        calls = []

        def on_start(**kwargs):
            calls.append(("start", kwargs))

        hook = EventHook(on_job_start=on_start)
        assert hook.on_job_start is on_start
        hook.on_job_start(job_id="123", job_name="test")
        assert len(calls) == 1
        assert calls[0] == ("start", {"job_id": "123", "job_name": "test"})


class TestHookRegistry:
    def test_register_hook(self):
        registry = HookRegistry()
        hook = EventHook()
        registry.register(hook)
        assert hook in registry.hooks

    def test_unregister_hook(self):
        registry = HookRegistry()
        hook = EventHook()
        registry.register(hook)
        registry.unregister(hook)
        assert hook not in registry.hooks

    def test_unregister_nonexistent_hook(self):
        registry = HookRegistry()
        hook = EventHook()
        # Should not raise
        registry.unregister(hook)
        assert len(registry.hooks) == 0

    def test_clear_hooks(self):
        registry = HookRegistry()
        registry.register(EventHook())
        registry.register(EventHook())
        registry.clear()
        assert len(registry.hooks) == 0

    def test_emit_calls_hook(self):
        registry = HookRegistry()
        calls = []

        hook = EventHook(on_job_start=lambda **kw: calls.append(kw))
        registry.register(hook)
        registry.emit("on_job_start", job_id="abc", job_name="test_job")

        assert len(calls) == 1
        assert calls[0]["job_id"] == "abc"
        assert calls[0]["job_name"] == "test_job"

    def test_emit_multiple_hooks(self):
        registry = HookRegistry()
        calls = []

        hook1 = EventHook(on_job_start=lambda **kw: calls.append(("hook1", kw)))
        hook2 = EventHook(on_job_start=lambda **kw: calls.append(("hook2", kw)))
        registry.register(hook1)
        registry.register(hook2)

        registry.emit("on_job_start", job_id="xyz")
        assert len(calls) == 2
        assert calls[0][0] == "hook1"
        assert calls[1][0] == "hook2"

    def test_emit_ignores_hooks_without_callback(self):
        registry = HookRegistry()
        calls = []

        hook_with = EventHook(on_job_start=lambda **kw: calls.append(kw))
        hook_without = EventHook()  # No callbacks
        registry.register(hook_with)
        registry.register(hook_without)

        registry.emit("on_job_start", job_id="123")
        assert len(calls) == 1

    def test_emit_handles_failing_hook(self, caplog):
        registry = HookRegistry()
        calls = []

        def failing_hook(**kw):
            raise ValueError("Hook error")

        def working_hook(**kw):
            calls.append(kw)

        hook1 = EventHook(on_job_start=failing_hook)
        hook2 = EventHook(on_job_start=working_hook)
        registry.register(hook1)
        registry.register(hook2)

        with caplog.at_level(logging.WARNING):
            registry.emit("on_job_start", job_id="test")

        # Second hook should still be called
        assert len(calls) == 1
        assert "Hook error" in caplog.text


class TestLoggingHook:
    def test_logging_hook_logs_job_start(self, caplog):
        hook = LoggingHook()
        with caplog.at_level(logging.INFO):
            hook.on_job_start(job_id="abc123", job_name="train_model")
        assert "Job started: train_model (abc123)" in caplog.text

    def test_logging_hook_logs_job_complete(self, caplog):
        hook = LoggingHook()
        with caplog.at_level(logging.INFO):
            hook.on_job_complete(job_id="abc123", job_name="train_model", runtime_seconds=42.5)
        assert "Job completed: train_model (abc123) in 42.50s" in caplog.text

    def test_logging_hook_logs_job_failed(self, caplog):
        hook = LoggingHook()
        with caplog.at_level(logging.ERROR):
            hook.on_job_failed(job_id="abc123", job_name="train_model", error="OOM")
        assert "Job failed: train_model (abc123): OOM" in caplog.text

    def test_logging_hook_custom_logger(self, caplog):
        hook = LoggingHook(logger_name="custom.logger")
        with caplog.at_level(logging.INFO, logger="custom.logger"):
            hook.on_job_start(job_id="x", job_name="y")
        assert "Job started: y (x)" in caplog.text


class TestGlobalHooksRegistry:
    def setup_method(self):
        hooks.clear()

    def teardown_method(self):
        hooks.clear()

    def test_global_hooks_registry_exists(self):
        assert isinstance(hooks, HookRegistry)

    def test_global_hooks_can_register(self):
        hook = EventHook()
        hooks.register(hook)
        assert hook in hooks.hooks
