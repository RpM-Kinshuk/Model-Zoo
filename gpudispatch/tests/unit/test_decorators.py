"""Tests for the @gpu decorator."""

import pytest
from unittest.mock import patch, MagicMock

from gpudispatch.decorators import gpu, set_default_dispatcher, get_default_dispatcher


class TestGPUDecorator:
    def test_decorator_basic(self):
        @gpu(count=1)
        def my_func():
            return 42

        assert callable(my_func)
        assert my_func.__name__ == "my_func"

    def test_decorator_preserves_docstring(self):
        @gpu(1)
        def documented_func():
            """This is my docstring."""
            return 42

        assert documented_func.__doc__ == "This is my docstring."

    def test_decorator_with_args(self):
        @gpu(2, memory="16GB")
        def train(lr, batch_size):
            return lr * batch_size

        assert train._gpu_count == 2
        assert train._gpu_memory == "16GB"

    def test_decorator_shorthand(self):
        @gpu(1)
        def simple_func():
            return 1

        assert simple_func._gpu_count == 1

    def test_decorator_no_parens(self):
        @gpu
        def simple():
            return 1

        assert simple._gpu_count == 1

    def test_set_default_dispatcher(self):
        mock_dispatcher = MagicMock()
        set_default_dispatcher(mock_dispatcher)
        assert get_default_dispatcher() is mock_dispatcher
        set_default_dispatcher(None)  # Clean up

    def test_decorator_without_dispatcher_warns(self):
        set_default_dispatcher(None)

        @gpu(1)
        def compute():
            return 42

        with pytest.warns(RuntimeWarning, match="No dispatcher set"):
            result = compute()

        assert result == 42  # Still runs locally


class TestGPUDecoratorWithDispatcher:
    """Tests for decorator behavior when dispatcher is set."""

    def setup_method(self):
        """Reset dispatcher before each test."""
        set_default_dispatcher(None)

    def teardown_method(self):
        """Clean up dispatcher after each test."""
        set_default_dispatcher(None)

    def test_decorator_submits_to_dispatcher(self):
        mock_dispatcher = MagicMock()
        mock_job = MagicMock()
        mock_dispatcher.submit.return_value = mock_job
        mock_dispatcher.wait.return_value = 42

        set_default_dispatcher(mock_dispatcher)

        @gpu(count=2, memory="8GB")
        def compute(x):
            return x * 2

        result = compute(10)

        mock_dispatcher.submit.assert_called_once()
        call_kwargs = mock_dispatcher.submit.call_args
        assert call_kwargs[1]["gpu"] == 2
        assert call_kwargs[1]["memory"] == "8GB"
        mock_dispatcher.wait.assert_called_once_with(mock_job)
        assert result == 42

    def test_decorator_passes_function_args(self):
        mock_dispatcher = MagicMock()
        mock_job = MagicMock()
        mock_dispatcher.submit.return_value = mock_job
        mock_dispatcher.wait.return_value = "result"

        set_default_dispatcher(mock_dispatcher)

        @gpu(1)
        def process(a, b, c=None):
            return f"{a}-{b}-{c}"

        process(1, 2, c="three")

        call_kwargs = mock_dispatcher.submit.call_args
        assert call_kwargs[1]["args"] == (1, 2)
        assert call_kwargs[1]["kwargs"] == {"c": "three"}

    def test_decorator_with_priority(self):
        mock_dispatcher = MagicMock()
        mock_dispatcher.submit.return_value = MagicMock()
        mock_dispatcher.wait.return_value = None

        set_default_dispatcher(mock_dispatcher)

        @gpu(1, priority=10)
        def high_priority_task():
            pass

        assert high_priority_task._gpu_priority == 10
        high_priority_task()

        call_kwargs = mock_dispatcher.submit.call_args
        assert call_kwargs[1]["priority"] == 10

    def test_original_function_accessible(self):
        @gpu(1)
        def my_function():
            """Original docstring."""
            return "original"

        assert hasattr(my_function, "_original_fn")
        assert my_function._original_fn() == "original"
