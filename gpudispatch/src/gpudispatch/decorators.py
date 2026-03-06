"""Decorators for GPU job dispatch."""

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, Optional, TypeVar, Union, overload

F = TypeVar("F", bound=Callable[..., Any])

_default_dispatcher: Optional[Any] = None


def set_default_dispatcher(dispatcher: Any) -> None:
    """Set the default dispatcher for @gpu decorated functions.

    Args:
        dispatcher: The Dispatcher instance to use for submitting jobs.
            Pass None to clear the default dispatcher.

    Example:
        >>> from gpudispatch.core import Dispatcher
        >>> from gpudispatch.decorators import set_default_dispatcher, gpu
        >>>
        >>> dispatcher = Dispatcher(gpus=[0, 1])
        >>> set_default_dispatcher(dispatcher)
        >>>
        >>> @gpu(2)
        ... def train_model():
        ...     pass
    """
    global _default_dispatcher
    _default_dispatcher = dispatcher


def get_default_dispatcher() -> Optional[Any]:
    """Get the current default dispatcher.

    Returns:
        The current default Dispatcher, or None if not set.
    """
    return _default_dispatcher


@overload
def gpu(fn: F) -> F:
    ...


@overload
def gpu(
    count: int = 1,
    *,
    memory: Optional[str] = None,
    priority: int = 0,
) -> Callable[[F], F]:
    ...


def gpu(
    fn: Optional[Union[F, int]] = None,
    count: int = 1,
    *,
    memory: Optional[str] = None,
    priority: int = 0,
) -> Union[F, Callable[[F], F]]:
    """Decorator to mark a function for GPU execution.

    Can be used in several ways:

        # Without parentheses (defaults to 1 GPU)
        @gpu
        def func(): ...

        # With just count as positional arg
        @gpu(2)
        def func(): ...

        # With named arguments
        @gpu(count=2, memory="16GB")
        def func(): ...

        # With count as first positional and other named args
        @gpu(2, memory="16GB", priority=10)
        def func(): ...

    Args:
        fn: Function to decorate (when used without parentheses).
        count: Number of GPUs required (default 1).
        memory: Memory requirement (e.g., "16GB").
        priority: Job priority (higher = more important, runs sooner).

    Returns:
        Decorated function that will be dispatched to available GPUs.

    Example:
        >>> @gpu(2, memory="16GB")
        ... def train(model, data):
        ...     return model.fit(data)
        >>>
        >>> # When called, the function is submitted to the dispatcher
        >>> result = train(my_model, my_data)
    """
    # Handle the case where @gpu is used without parentheses
    # In this case, fn will be the decorated function
    if callable(fn):
        return _create_decorator(count=1, memory=None, priority=0)(fn)

    # Handle the case where first arg is an integer (count)
    # e.g., @gpu(2) or @gpu(2, memory="8GB")
    if isinstance(fn, int):
        return _create_decorator(count=fn, memory=memory, priority=priority)

    # Handle the case where fn is None (using keyword args only)
    # e.g., @gpu(count=2, memory="8GB")
    return _create_decorator(count=count, memory=memory, priority=priority)


def _create_decorator(
    count: int,
    memory: Optional[str],
    priority: int,
) -> Callable[[F], F]:
    """Create the actual decorator with the given parameters.

    Args:
        count: Number of GPUs required.
        memory: Memory requirement string.
        priority: Job priority.

    Returns:
        A decorator function.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            dispatcher = get_default_dispatcher()

            if dispatcher is None:
                warnings.warn(
                    "No dispatcher set. Running function locally. "
                    "Use set_default_dispatcher() or Dispatcher context manager.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return func(*args, **kwargs)

            # Submit job to dispatcher
            job = dispatcher.submit(
                func,
                args=args,
                kwargs=kwargs,
                gpu=count,
                memory=memory,
                priority=priority,
            )

            # Wait for job completion and return result
            return dispatcher.wait(job)

        # Attach metadata to the wrapper for introspection
        wrapper._gpu_count = count  # type: ignore[attr-defined]
        wrapper._gpu_memory = memory  # type: ignore[attr-defined]
        wrapper._gpu_priority = priority  # type: ignore[attr-defined]
        wrapper._original_fn = func  # type: ignore[attr-defined]

        return wrapper  # type: ignore[return-value]

    return decorator
