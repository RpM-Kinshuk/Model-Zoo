# Future Scalability: Supervisor Loop

The current scheduler uses one monitor thread per active worker. This is simple
and appropriate while active concurrency is bounded by GPU count. If the infra
ever needs to supervise hundreds or thousands of concurrent workers, move to a
single supervisor loop.

## Goal

Keep the same worker contract while reducing per-worker thread overhead.

Workers still expose:

- process ID / process group ID
- heartbeat file
- `stage`
- `stage_entered_at`
- terminal status record
- optional per-worker cache path

## Current Shape

dispatcher
  -> worker process
  -> monitor thread per worker
  -> heartbeat file per worker

This scales with active workers. It is fine for GPU-bound runs with small
concurrency.

## Scaled Shape

dispatcher
  -> worker processes
  -> one supervisor loop
       -> polls process exits
       -> scans heartbeat files
       -> applies heartbeat/stage timeouts
       -> terminates stale process groups
       -> updates current_state.json

## Design

Maintain one in-memory table of active workers:

worker_id -> {
  pid,
  pgid,
  assigned_gpus,
  heartbeat_path,
  log_path,
  cache_path,
  started_at,
  last_status,
}

The supervisor loop wakes on a fixed interval and performs one batch pass:
1. Poll all active processes for exit.
2. Read heartbeat metadata for active workers.
3. Apply heartbeat timeout.
4. Apply stage timeout.
5. Log or terminate stale workers according to config.
6. Cleanup finished workers.
7. Rewrite current_state.json.

## What Stays The Same

- heartbeat semantics
- stage timeout semantics
- gpu_config.json runtime reload
- process-group termination
- terminal status records
- per-worker cache cleanup
- worker implementation contract

## What Changes

- no monitor thread per worker
- stale checks are centralized
- process polling is batched
- active state is owned by the supervisor loop

## When To Do This

Do this only if active worker concurrency grows into the hundreds. For GPU-
bound ESD runs, where concurrency is normally limited by GPU count, the current
design is simpler and sufficient.