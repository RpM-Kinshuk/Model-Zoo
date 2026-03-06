"""Tests for search space primitives."""

import math
import random
from typing import List

import pytest

from gpudispatch.experiments.search_space import (
    Choice,
    Distribution,
    Grid,
    Int,
    Log,
    Range,
    SearchSpace,
    Sweep,
    Uniform,
)


class TestDistributionBase:
    """Test the Distribution ABC."""

    def test_distribution_is_abstract(self):
        """Distribution cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Distribution()  # type: ignore


class TestChoice:
    """Tests for Choice distribution."""

    def test_choice_sample_returns_element(self):
        """sample() returns one of the provided choices."""
        c = Choice([16, 32, 64])
        for _ in range(100):
            sample = c.sample()
            assert sample in [16, 32, 64]

    def test_choice_contains(self):
        """contains() checks membership."""
        c = Choice(["a", "b", "c"])
        assert c.contains("a")
        assert c.contains("b")
        assert not c.contains("d")

    def test_choice_single_element(self):
        """Single-element choice always returns that element."""
        c = Choice([42])
        for _ in range(10):
            assert c.sample() == 42

    def test_choice_values_property(self):
        """values property returns the choices."""
        c = Choice([1, 2, 3])
        assert c.values == [1, 2, 3]

    def test_choice_len(self):
        """len() returns number of choices."""
        c = Choice([1, 2, 3, 4])
        assert len(c) == 4

    def test_choice_iter(self):
        """Choice is iterable."""
        c = Choice(["x", "y", "z"])
        assert list(c) == ["x", "y", "z"]


class TestLog:
    """Tests for Log (log-uniform) distribution."""

    def test_log_sample_in_bounds(self):
        """sample() returns value in [low, high]."""
        log_dist = Log(1e-5, 1e-1)
        for _ in range(100):
            sample = log_dist.sample()
            assert 1e-5 <= sample <= 1e-1

    def test_log_contains_in_bounds(self):
        """contains() returns True for values in range."""
        log_dist = Log(1e-4, 1e-2)
        assert log_dist.contains(1e-3)
        assert log_dist.contains(1e-4)  # Boundary
        assert log_dist.contains(1e-2)  # Boundary

    def test_log_contains_out_of_bounds(self):
        """contains() returns False for values outside range."""
        log_dist = Log(1e-4, 1e-2)
        assert not log_dist.contains(1e-5)
        assert not log_dist.contains(1e-1)

    def test_log_distribution_is_log_scale(self):
        """Sample distribution should be roughly log-uniform."""
        random.seed(42)
        log_dist = Log(1e-4, 1e-0)
        samples = [log_dist.sample() for _ in range(1000)]

        # In log-uniform, roughly equal samples in each decade
        in_4_3 = sum(1 for s in samples if 1e-4 <= s < 1e-3)
        in_3_2 = sum(1 for s in samples if 1e-3 <= s < 1e-2)
        in_2_1 = sum(1 for s in samples if 1e-2 <= s < 1e-1)
        in_1_0 = sum(1 for s in samples if 1e-1 <= s <= 1e-0)

        # Each decade should have roughly 25% (with some tolerance)
        for count in [in_4_3, in_3_2, in_2_1, in_1_0]:
            assert 150 < count < 350, f"Expected ~250, got {count}"

    def test_log_low_high_properties(self):
        """low and high properties work."""
        log_dist = Log(0.001, 0.1)
        assert log_dist.low == 0.001
        assert log_dist.high == 0.1


class TestUniform:
    """Tests for Uniform distribution."""

    def test_uniform_sample_in_bounds(self):
        """sample() returns value in [low, high]."""
        u = Uniform(0.0, 0.5)
        for _ in range(100):
            sample = u.sample()
            assert 0.0 <= sample <= 0.5

    def test_uniform_contains(self):
        """contains() checks range."""
        u = Uniform(0.0, 1.0)
        assert u.contains(0.0)
        assert u.contains(0.5)
        assert u.contains(1.0)
        assert not u.contains(-0.1)
        assert not u.contains(1.1)

    def test_uniform_low_high_properties(self):
        """low and high properties work."""
        u = Uniform(0.1, 0.9)
        assert u.low == 0.1
        assert u.high == 0.9


class TestInt:
    """Tests for Int distribution."""

    def test_int_sample_in_bounds(self):
        """sample() returns integer in [low, high]."""
        i = Int(4, 32)
        for _ in range(100):
            sample = i.sample()
            assert 4 <= sample <= 32
            assert isinstance(sample, int)

    def test_int_contains(self):
        """contains() checks integer range."""
        i = Int(1, 10)
        assert i.contains(1)
        assert i.contains(5)
        assert i.contains(10)
        assert not i.contains(0)
        assert not i.contains(11)

    def test_int_contains_rejects_non_int(self):
        """contains() returns False for non-integer values."""
        i = Int(1, 10)
        assert not i.contains(5.5)

    def test_int_inclusive_bounds(self):
        """Both low and high are inclusive."""
        i = Int(1, 1)
        assert i.sample() == 1
        assert i.contains(1)


class TestRange:
    """Tests for Range (stepped range like arange but inclusive)."""

    def test_range_values(self):
        """Range generates correct values."""
        r = Range(0.1, 0.3, 0.1)
        values = list(r)
        assert len(values) == 3
        assert math.isclose(values[0], 0.1)
        assert math.isclose(values[1], 0.2)
        assert math.isclose(values[2], 0.3)

    def test_range_sample(self):
        """sample() returns one of the range values."""
        r = Range(0.1, 1.0, 0.1)
        values = list(r)
        for _ in range(100):
            sample = r.sample()
            assert any(math.isclose(sample, v) for v in values)

    def test_range_contains(self):
        """contains() checks if value is in range."""
        r = Range(0.0, 1.0, 0.25)
        assert r.contains(0.0)
        assert r.contains(0.25)
        assert r.contains(0.5)
        assert r.contains(0.75)
        assert r.contains(1.0)
        assert not r.contains(0.1)

    def test_range_len(self):
        """len() returns number of values."""
        r = Range(0, 10, 2)  # 0, 2, 4, 6, 8, 10
        assert len(r) == 6


class TestGrid:
    """Tests for Grid (exhaustive combinations)."""

    def test_grid_iteration(self):
        """Grid iterates over all combinations."""
        g = Grid(lr=[1e-4, 1e-3], batch=[16, 32])
        combinations = list(g)

        assert len(combinations) == 4
        assert {"lr": 1e-4, "batch": 16} in combinations
        assert {"lr": 1e-4, "batch": 32} in combinations
        assert {"lr": 1e-3, "batch": 16} in combinations
        assert {"lr": 1e-3, "batch": 32} in combinations

    def test_grid_size(self):
        """size property returns total combinations."""
        g = Grid(a=[1, 2], b=[3, 4], c=[5, 6])
        assert g.size == 8  # 2 * 2 * 2

    def test_grid_len(self):
        """len() returns size."""
        g = Grid(x=[1, 2, 3], y=["a", "b"])
        assert len(g) == 6

    def test_grid_single_param(self):
        """Grid with single parameter."""
        g = Grid(model=["small", "medium", "large"])
        combinations = list(g)
        assert len(combinations) == 3
        assert {"model": "small"} in combinations

    def test_grid_empty(self):
        """Empty grid has size 0."""
        g = Grid()
        assert g.size == 0
        assert list(g) == []

    def test_grid_parameters(self):
        """parameters property returns parameter names."""
        g = Grid(lr=[1, 2], batch=[16])
        assert set(g.parameters) == {"lr", "batch"}


class TestSweep:
    """Tests for Sweep (sampled distributions)."""

    def test_sweep_sample(self):
        """sample() returns dict with sampled values."""
        s = Sweep(lr=Log(1e-5, 1e-1), dropout=Uniform(0.0, 0.5))
        sample = s.sample()

        assert "lr" in sample
        assert "dropout" in sample
        assert 1e-5 <= sample["lr"] <= 1e-1
        assert 0.0 <= sample["dropout"] <= 0.5

    def test_sweep_sample_choice(self):
        """Sweep works with Choice distribution."""
        s = Sweep(activation=Choice(["relu", "gelu", "tanh"]))
        sample = s.sample()
        assert sample["activation"] in ["relu", "gelu", "tanh"]

    def test_sweep_parameters(self):
        """parameters property returns parameter names."""
        s = Sweep(lr=Log(1e-5, 1e-1), batch=Int(16, 64))
        assert set(s.parameters) == {"lr", "batch"}

    def test_sweep_distributions(self):
        """distributions property returns the distributions."""
        lr_dist = Log(1e-5, 1e-1)
        s = Sweep(lr=lr_dist)
        assert s.distributions["lr"] is lr_dist


class TestSearchSpace:
    """Tests for SearchSpace (combined grid + sweep)."""

    def test_search_space_from_dict_grid_only(self):
        """SearchSpace with only grid parameters."""
        space = SearchSpace.from_dict({
            "model": ["small", "large"],
            "optimizer": ["adam", "sgd"],
        })
        assert space.is_grid
        assert space.has_grid
        assert not space.has_sweep

    def test_search_space_from_dict_sweep_only(self):
        """SearchSpace with only sweep parameters."""
        space = SearchSpace.from_dict({
            "lr": Log(1e-5, 1e-1),
            "dropout": Uniform(0.0, 0.5),
        })
        assert not space.is_grid
        assert not space.has_grid
        assert space.has_sweep

    def test_search_space_from_dict_mixed(self):
        """SearchSpace with both grid and sweep."""
        space = SearchSpace.from_dict({
            "model": ["small", "large"],
            "lr": Log(1e-5, 1e-1),
        })
        assert not space.is_grid
        assert space.has_grid
        assert space.has_sweep

    def test_search_space_grid_size(self):
        """grid_size returns size of grid component."""
        space = SearchSpace.from_dict({
            "model": ["small", "large"],
            "batch": [16, 32, 64],
        })
        assert space.grid_size == 6

    def test_search_space_sample(self):
        """sample() combines grid point with sweep sample."""
        space = SearchSpace.from_dict({
            "model": ["small"],
            "lr": Log(1e-5, 1e-1),
        })
        sample = space.sample()
        assert sample["model"] == "small"
        assert 1e-5 <= sample["lr"] <= 1e-1

    def test_search_space_parameters(self):
        """parameters property returns all parameter names."""
        space = SearchSpace.from_dict({
            "model": ["small", "large"],
            "lr": Log(1e-5, 1e-1),
        })
        assert set(space.parameters) == {"model", "lr"}

    def test_search_space_iter_grid(self):
        """iter_grid() iterates over grid points."""
        space = SearchSpace.from_dict({
            "model": ["a", "b"],
            "batch": [16],
        })
        grid_points = list(space.iter_grid())
        assert len(grid_points) == 2
        assert {"model": "a", "batch": 16} in grid_points
        assert {"model": "b", "batch": 16} in grid_points
