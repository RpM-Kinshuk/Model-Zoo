# GPU And Worker Supervision

Workers are launched with `CUDA_VISIBLE_DEVICES` set by `gputracker`. Inside a worker, the assigned physical GPU appears as local `cuda:0` when one GPU is assigned.

Current checks:

- GPU availability: `gputracker` polls memory usage before dispatch.
- Worker heartbeat: catches dead or blocked heartbeat writers.
- Stage timeout: catches foreground work stuck in a stage such as `load`, even if the heartbeat thread is still alive.
- Process groups: each worker runs in its own PGID so terminate/kill actions stay scoped to that worker.
- Active state: `<output_dir>/logs/current_state.json` records PID, PGID, assigned GPUs, stage, heartbeat, log path, and cache path.

Runtime knobs in `<output_dir>/gpu_config.json`:

```json
{
  "stale_process_action": "log",
  "heartbeat_timeout_seconds": 7200,
  "stage_timeout_seconds": {
    "load": 7200,
    "analyze": 28800,
    "save": 1800,
    "default": 14400
  },
  "termination_grace_seconds": 30
}
```

Use `log` while calibrating. Use `terminate` after the timeout windows are trusted.

To reload config:

```bash
kill -HUP <runner_pid>
```
