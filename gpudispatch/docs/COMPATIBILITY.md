# Compatibility Matrix

This document tracks what `gpudispatch` supports today, what CI verifies on every change,
and what remains best-effort.

## Versioning Policy

- Project versioning follows SemVer (`MAJOR.MINOR.PATCH`).
- Breaking API changes must go into a major version bump.
- Deprecations should be documented in release notes before removal.

## Python Compatibility

| Python | Support Status | CI Coverage |
|---|---|---|
| 3.9 | Supported | Full unit suite |
| 3.10 | Supported | Full unit suite |
| 3.11 | Supported | Full unit suite + package build smoke |
| 3.12 | Supported | Full unit suite |
| < 3.9 | Not supported | Not tested |

Source of truth:
- `requires-python = ">=3.9"` in `pyproject.toml`
- CI matrix in `.github/workflows/gpudispatch-ci.yml`

## Backend Compatibility

| Backend | Status | Notes |
|---|---|---|
| LocalBackend | Supported | Single-machine orchestration |
| SLURMBackend | Supported | Designed for SLURM allocations and scheduler command integration |
| Kubernetes | Planned | Not implemented yet |
| AWS | Planned | Not implemented yet |
| GCP | Planned | Not implemented yet |

## Platform Expectations

| Platform | Status | Notes |
|---|---|---|
| Linux | Primary target | Most GPU/cluster workflows expected here |
| macOS | Best effort | Useful for development/control-plane logic |
| Windows | Best effort | Useful for development/control-plane logic |

## CI and Release Automation

- CI workflow: `.github/workflows/gpudispatch-ci.yml`
  - Runs `tests/unit` for Python 3.9–3.12.
  - Builds wheel/sdist and validates metadata via `twine check`.
- Release workflow: `.github/workflows/gpudispatch-release.yml`
  - Publishes from tags matching `gpudispatch-v*`.
  - Supports manual publish to TestPyPI or PyPI.

## Maintainer Checklist for Compatibility Updates

1. Update `pyproject.toml` classifiers and `requires-python` if support changes.
2. Update CI matrix in `.github/workflows/gpudispatch-ci.yml`.
3. Update this matrix doc.
4. Mention the change in release notes.
