# Contributor Guide: Plugin Authors

`gpudispatch` supports extension points that can be implemented in separate packages.
This guide focuses on practical plugin-style integrations.

## Plugin Types

### 1) Observability Hook Plugins

Implement custom telemetry/exporters by subclassing or configuring `EventHook`.

Common use cases:

- Prometheus metrics export
- OpenTelemetry trace export
- Alerting integrations (Slack, PagerDuty)

```python
from gpudispatch.observability.hooks import EventHook, hooks


def _job_failed(job_id: str, job_name: str, error: str, **kw) -> None:
    print(f"ALERT: {job_name} ({job_id}) failed: {error}")


hooks.register(EventHook(on_job_failed=_job_failed))
```

### 2) Storage Plugins

Implement custom experiment persistence by implementing `Storage`.

- Interface: `gpudispatch.experiments.storage.base.Storage`
- Pass the implementation to `Experiment(storage=...)`

### 3) Search/Optimization Plugins

Implement custom trial proposal logic by implementing `Strategy`.

- Interface: `gpudispatch.experiments.strategies.base.Strategy`
- Pass into `Experiment(strategy=...)`

## Integration Pattern

`gpudispatch` currently uses explicit registration/injection, not dynamic plugin discovery.

- Hooks: register with global `hooks`
- Storage: constructor injection to `Experiment`
- Strategy: constructor injection to `Experiment`

This explicit model keeps behavior transparent and easy to debug.

## Compatibility Expectations

When shipping an external plugin package:

1. Pin compatible `gpudispatch` versions in your package metadata.
2. Test against all supported Python versions listed in `docs/COMPATIBILITY.md`.
3. Treat hook payloads as additive (new keys may be introduced over time).
4. Keep handlers tolerant of unknown keyword arguments.

## Testing Recommendations

- Unit-test your plugin behavior in isolation.
- Add an integration test with real `gpudispatch` objects.
- Ensure failure in plugin code does not crash orchestration flows.

`HookRegistry.emit()` already isolates hook failures and logs warnings, but plugin code
should still handle network and serialization errors defensively.
