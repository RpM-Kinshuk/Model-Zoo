"""Tests for the reproducibility module.

This module tests:
- Capturing reproducibility context (Python version, timestamp, git commit)
- Setting random seeds for reproducibility
- Edge cases and error handling
"""

from __future__ import annotations

import random
import sys
from datetime import datetime
from typing import Any, Dict
from unittest.mock import patch

import pytest

from gpudispatch.experiments.reproducibility import (
    capture_context,
    set_seeds,
    _get_git_commit,
)


class TestCaptureContext:
    """Tests for capture_context function."""

    def test_returns_dict(self) -> None:
        """Test that capture_context returns a dictionary."""
        context = capture_context()

        assert isinstance(context, dict)

    def test_includes_python_version(self) -> None:
        """Test that context includes Python version."""
        context = capture_context()

        assert "python_version" in context
        assert context["python_version"] == sys.version

    def test_includes_timestamp(self) -> None:
        """Test that context includes ISO format timestamp."""
        before = datetime.now()
        context = capture_context()
        after = datetime.now()

        assert "timestamp" in context
        # Should be valid ISO format
        timestamp = datetime.fromisoformat(context["timestamp"])
        assert before <= timestamp <= after

    def test_includes_git_commit(self) -> None:
        """Test that context includes git commit hash (may be None)."""
        context = capture_context()

        assert "git_commit" in context
        # git_commit can be None if not in a git repo
        if context["git_commit"] is not None:
            # Should be a valid hex string (40 chars for full SHA)
            assert isinstance(context["git_commit"], str)
            assert len(context["git_commit"]) > 0

    def test_includes_random_seed_placeholder(self) -> None:
        """Test that context includes random_seed placeholder."""
        context = capture_context()

        assert "random_seed" in context
        # Placeholder is None by default
        assert context["random_seed"] is None

    def test_captures_user_seed_if_set(self) -> None:
        """Test that context captures seed if set by user before capture."""
        set_seeds(42)
        context = capture_context(seed=42)

        assert context["random_seed"] == 42


class TestGetGitCommit:
    """Tests for _get_git_commit helper function."""

    def test_returns_string_or_none(self) -> None:
        """Test that _get_git_commit returns a string or None."""
        result = _get_git_commit()

        assert result is None or isinstance(result, str)

    @patch("subprocess.run")
    def test_returns_none_on_subprocess_error(self, mock_run: Any) -> None:
        """Test that _get_git_commit returns None on subprocess error."""
        mock_run.side_effect = FileNotFoundError("git not found")

        result = _get_git_commit()

        assert result is None

    @patch("subprocess.run")
    def test_returns_none_on_non_zero_return_code(self, mock_run: Any) -> None:
        """Test that _get_git_commit returns None when git returns non-zero."""
        mock_run.return_value.returncode = 128
        mock_run.return_value.stdout = ""

        result = _get_git_commit()

        assert result is None

    @patch("subprocess.run")
    def test_returns_commit_hash_on_success(self, mock_run: Any) -> None:
        """Test that _get_git_commit returns commit hash on success."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "abc123def456\n"

        result = _get_git_commit()

        assert result == "abc123def456"


class TestSetSeeds:
    """Tests for set_seeds function."""

    def test_sets_python_random_seed(self) -> None:
        """Test that set_seeds sets Python random module seed."""
        set_seeds(42)
        val1 = random.random()

        set_seeds(42)
        val2 = random.random()

        assert val1 == val2

    def test_different_seeds_produce_different_values(self) -> None:
        """Test that different seeds produce different random sequences."""
        set_seeds(42)
        val1 = random.random()

        set_seeds(123)
        val2 = random.random()

        assert val1 != val2

    def test_sets_numpy_seed_if_available(self) -> None:
        """Test that set_seeds sets numpy seed if numpy is available."""
        try:
            import numpy as np

            set_seeds(42)
            val1 = np.random.random()

            set_seeds(42)
            val2 = np.random.random()

            assert val1 == val2
        except ImportError:
            pytest.skip("numpy not available")

    def test_sets_torch_seed_if_available(self) -> None:
        """Test that set_seeds sets torch seed if torch is available."""
        try:
            import torch

            set_seeds(42)
            val1 = torch.rand(1).item()

            set_seeds(42)
            val2 = torch.rand(1).item()

            assert val1 == val2
        except ImportError:
            pytest.skip("torch not available")

    def test_accepts_integer_seed(self) -> None:
        """Test that set_seeds accepts integer seeds."""
        # Should not raise
        set_seeds(0)
        set_seeds(2**31 - 1)
        set_seeds(12345)
