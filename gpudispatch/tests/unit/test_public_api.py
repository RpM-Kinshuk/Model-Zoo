"""Tests for gpudispatch top-level public API exports."""

from gpudispatch import (
    CommandResult,
    Dispatcher,
    DispatcherProfile,
    dispatcher_from_profile,
    get_profile,
    list_profiles,
)
from gpudispatch.core import CommandResult as CoreCommandResult
from gpudispatch.core import Dispatcher as CoreDispatcher
from gpudispatch.profiles import (
    DispatcherProfile as CoreDispatcherProfile,
    dispatcher_from_profile as core_dispatcher_from_profile,
    get_profile as core_get_profile,
    list_profiles as core_list_profiles,
)


def test_dispatcher_exported_from_top_level() -> None:
    assert Dispatcher is CoreDispatcher


def test_command_result_exported_from_top_level() -> None:
    assert CommandResult is CoreCommandResult


def test_profiles_api_exported_from_top_level() -> None:
    assert DispatcherProfile is CoreDispatcherProfile
    assert dispatcher_from_profile is core_dispatcher_from_profile
    assert get_profile is core_get_profile
    assert list_profiles is core_list_profiles
