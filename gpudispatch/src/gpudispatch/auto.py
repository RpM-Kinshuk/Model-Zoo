"""Auto-detection and backend selection for GPU orchestration.

This module provides automatic environment detection and backend selection,
allowing gpudispatch to work seamlessly across different compute environments
(local machines, SLURM clusters, Kubernetes, cloud providers, etc.).

Example:
    >>> from gpudispatch import auto_dispatcher
    >>> # Automatically detects environment and selects appropriate backend
    >>> dispatcher = auto_dispatcher()  # Just works everywhere
    >>> with dispatcher:
    ...     gpus = dispatcher.allocate_gpus(2)
    ...     # use GPUs
    ...     dispatcher.release_gpus(gpus)
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Any, List, Optional, Union

from gpudispatch.backends.base import Backend
from gpudispatch.backends.local import LocalBackend

logger = logging.getLogger(__name__)


class EnvironmentType(Enum):
    """Detected compute environment type.

    Used internally to determine which backend to instantiate.
    """

    LOCAL = "local"
    SLURM = "slurm"
    KUBERNETES = "kubernetes"
    AWS = "aws"
    GCP = "gcp"


def detect_environment() -> EnvironmentType:
    """Detect the current compute environment.

    Detection priority:
    1. SLURM cluster (SLURM_JOB_ID or SLURM_NODELIST env vars)
    2. Kubernetes (KUBERNETES_SERVICE_HOST or KUBERNETES_SERVICE_PORT env vars)
    3. AWS (future: instance metadata endpoint)
    4. GCP (future: instance metadata endpoint)
    5. Local (default fallback)

    Returns:
        EnvironmentType: The detected environment type.
    """
    # Check for SLURM environment
    if os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_NODELIST"):
        logger.debug("Detected SLURM environment")
        return EnvironmentType.SLURM

    # Check for Kubernetes environment
    if os.environ.get("KUBERNETES_SERVICE_HOST") or os.environ.get(
        "KUBERNETES_SERVICE_PORT"
    ):
        logger.debug("Detected Kubernetes environment")
        return EnvironmentType.KUBERNETES

    # TODO: Add AWS detection via instance metadata endpoint
    # TODO: Add GCP detection via instance metadata endpoint

    # Default to local
    logger.debug("Defaulting to local environment")
    return EnvironmentType.LOCAL


def auto_dispatcher(
    force_backend: Optional[str] = None,
    gpus: Union[str, List[int]] = "auto",
    memory_threshold: Union[str, int] = "500MB",
    polling_interval: int = 5,
    process_mode: str = "subprocess",
    **kwargs: Any,
) -> Backend:
    """Automatically detect environment and create appropriate backend.

    This is the recommended way to create a dispatcher when you want your
    code to work across different compute environments without modification.

    Detection methods:
    1. SLURM cluster: Check for SLURM_JOB_ID env var
    2. Kubernetes: Check for KUBERNETES_SERVICE_HOST env var
    3. AWS/GCP: Check instance metadata endpoints (future)
    4. Default: LocalBackend for single-machine workloads

    Note: In Phase 2, only LocalBackend is fully implemented. Other backends
    will raise NotImplementedError with a helpful message.

    Args:
        force_backend: Force a specific backend type instead of auto-detection.
            Valid values: "local", "slurm", "kubernetes", "aws", "gcp".
            If None (default), auto-detection is used.
        gpus: GPU configuration for LocalBackend. Either "auto" for automatic
            detection, or a list of GPU indices. Default: "auto".
        memory_threshold: Memory threshold for LocalBackend. Consider a GPU
            "free" if used memory is below this. Default: "500MB".
        polling_interval: Seconds between availability checks. Default: 5.
        process_mode: Execution mode ("subprocess" or "thread"). Default: "subprocess".
        **kwargs: Additional keyword arguments passed to the backend.

    Returns:
        Backend: An appropriate backend instance for the detected environment.

    Raises:
        NotImplementedError: If the detected (or forced) environment is not
            yet supported (SLURM, Kubernetes, AWS, GCP).
        ValueError: If force_backend specifies an unknown backend type.

    Example:
        >>> from gpudispatch import auto_dispatcher
        >>>
        >>> # Auto-detect and create dispatcher
        >>> dispatcher = auto_dispatcher()
        >>>
        >>> # Force local backend with custom config
        >>> dispatcher = auto_dispatcher(
        ...     force_backend="local",
        ...     gpus=[0, 1],
        ...     memory_threshold="1GB",
        ... )
    """
    # Determine environment type
    if force_backend is not None:
        env_type = _parse_backend_name(force_backend)
    else:
        try:
            env_type = detect_environment()
        except Exception as e:
            logger.warning(
                f"Environment detection failed: {e}. Falling back to local backend."
            )
            env_type = EnvironmentType.LOCAL

    # Create appropriate backend
    return _create_backend(
        env_type=env_type,
        gpus=gpus,
        memory_threshold=memory_threshold,
        polling_interval=polling_interval,
        process_mode=process_mode,
        **kwargs,
    )


def _parse_backend_name(name: str) -> EnvironmentType:
    """Parse backend name string to EnvironmentType.

    Args:
        name: Backend name (case-insensitive).

    Returns:
        EnvironmentType: The corresponding environment type.

    Raises:
        ValueError: If the name is not recognized.
    """
    name_lower = name.lower().strip()

    mapping = {
        "local": EnvironmentType.LOCAL,
        "slurm": EnvironmentType.SLURM,
        "kubernetes": EnvironmentType.KUBERNETES,
        "k8s": EnvironmentType.KUBERNETES,
        "aws": EnvironmentType.AWS,
        "gcp": EnvironmentType.GCP,
        "google": EnvironmentType.GCP,
    }

    if name_lower not in mapping:
        valid_options = ", ".join(sorted(set(mapping.keys())))
        raise ValueError(
            f"Unknown backend '{name}'. Valid options: {valid_options}"
        )

    return mapping[name_lower]


def _create_backend(
    env_type: EnvironmentType,
    gpus: Union[str, List[int]] = "auto",
    memory_threshold: Union[str, int] = "500MB",
    polling_interval: int = 5,
    process_mode: str = "subprocess",
    **kwargs: Any,
) -> Backend:
    """Create a backend instance for the given environment type.

    Args:
        env_type: The environment type to create a backend for.
        gpus: GPU configuration (for LocalBackend).
        memory_threshold: Memory threshold (for LocalBackend).
        polling_interval: Polling interval (for LocalBackend).
        process_mode: Process mode (for LocalBackend).
        **kwargs: Additional backend-specific arguments.

    Returns:
        Backend: The created backend instance.

    Raises:
        NotImplementedError: If the environment is not yet supported.
    """
    if env_type == EnvironmentType.LOCAL:
        logger.info("Creating LocalBackend")
        return LocalBackend(
            gpus=gpus,
            memory_threshold=memory_threshold,
            polling_interval=polling_interval,
            process_mode=process_mode,
        )

    elif env_type == EnvironmentType.SLURM:
        raise NotImplementedError(
            "SLURM backend is not yet implemented. "
            "This will be available in a future release. "
            "For now, you can use LocalBackend on SLURM compute nodes by "
            "setting force_backend='local' in auto_dispatcher()."
        )

    elif env_type == EnvironmentType.KUBERNETES:
        raise NotImplementedError(
            "Kubernetes (K8s) backend is not yet implemented. "
            "This will be available in a future release. "
            "For now, you can use LocalBackend in Kubernetes pods by "
            "setting force_backend='local' in auto_dispatcher()."
        )

    elif env_type == EnvironmentType.AWS:
        raise NotImplementedError(
            "AWS backend is not yet implemented. "
            "This will be available in a future release. "
            "For now, you can use LocalBackend on AWS instances by "
            "setting force_backend='local' in auto_dispatcher()."
        )

    elif env_type == EnvironmentType.GCP:
        raise NotImplementedError(
            "GCP backend is not yet implemented. "
            "This will be available in a future release. "
            "For now, you can use LocalBackend on GCP instances by "
            "setting force_backend='local' in auto_dispatcher()."
        )

    else:
        # Fallback for any unknown environment type
        logger.warning(f"Unknown environment type {env_type}, using LocalBackend")
        return LocalBackend(
            gpus=gpus,
            memory_threshold=memory_threshold,
            polling_interval=polling_interval,
            process_mode=process_mode,
        )
