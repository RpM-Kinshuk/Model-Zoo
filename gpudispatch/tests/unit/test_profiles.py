"""Tests for opinionated dispatcher profiles."""

import pytest

from gpudispatch.core import Dispatcher
from gpudispatch.profiles import dispatcher_from_profile, get_profile, list_profiles


class TestListProfiles:
    def test_list_profiles_contains_expected_presets(self):
        profiles = list_profiles()
        assert set(profiles) == {"quickstart", "batch", "high_reliability"}


class TestGetProfile:
    def test_get_profile_is_case_insensitive(self):
        profile = get_profile("QuickStart")
        assert profile.name == "quickstart"

    def test_get_profile_raises_for_unknown_name(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            get_profile("does-not-exist")


class TestDispatcherFromProfile:
    def test_profile_dispatcher_applies_preset_defaults(self):
        dispatcher = dispatcher_from_profile("batch", gpus=[0])

        assert isinstance(dispatcher, Dispatcher)
        assert dispatcher.memory_threshold_mb == 1024
        assert dispatcher._default_command_timeout == 6 * 60 * 60
        assert dispatcher._default_command_env["PYTHONUNBUFFERED"] == "1"

    def test_profile_dispatcher_allows_explicit_overrides(self):
        dispatcher = dispatcher_from_profile(
            "high_reliability",
            gpus=[0],
            polling_interval=0.25,
            default_command_timeout=60,
            default_command_env={"CUSTOM": "yes"},
        )

        assert dispatcher._polling_interval == 0.25
        assert dispatcher._default_command_timeout == 60
        assert dispatcher._default_command_env == {"CUSTOM": "yes"}
