"""A guided terminal launcher for Model-Zoo's collection workflows."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import glob
import getpass
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
from typing import Sequence


from zoo.progress import Progress
from zoo.terminal_style import PageLayout, Theme, cell_width, wrap_cells

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUERY_ROOT = PROJECT_ROOT.parent / "Zoo-Keeper"
VERBOSE = False
MOTION = True


THEME = Theme()


def _clear() -> None:
    if sys.stdout.isatty() and _supports_full_screen():
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.flush()


def _layout() -> PageLayout:
    return PageLayout(shutil.get_terminal_size((84, 24)).columns)


def _write(text: str, role: str = "text", *, wrap: bool = True, literal: bool = False,
           flush: bool = False) -> None:
    layout = _layout()
    lines = wrap_cells(text, layout.width, literal=literal) if wrap else [_fit(text, layout.width)]
    for line in lines:
        print(layout.indent + THEME.paint(line, role), flush=flush)


def _panel(lines: Sequence[str], width: int | None = None, *, literal: bool = False) -> None:
    layout = _layout()
    width = min(width or layout.width, layout.width)
    if width < 8:
        for line in lines:
            _write(line, literal=literal)
        return
    inside = width - 4
    edge = THEME.paint("│", "border")
    print(layout.indent + THEME.paint("╭" + "─" * (width - 2) + "╮", "border"))
    for line in lines:
        for content in wrap_cells(line, inside, literal=literal):
            padding = " " * (inside - cell_width(content))
            print(f"{layout.indent}{edge} {content}{padding} {edge}")
    print(layout.indent + THEME.paint("╰" + "─" * (width - 2) + "╯", "border"))


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _completion_matches(buffer: str, completion: Sequence[str] | str | None) -> list[str]:
    if completion == "multi_path":
        boundary = buffer.rfind(";") + 1
        prefix, fragment = buffer[:boundary], buffer[boundary:].lstrip()
        spacing = buffer[boundary:]
        leading = spacing[:len(spacing) - len(spacing.lstrip())]
        return [prefix + leading + match for match in _completion_matches(fragment, "path")]
    if completion == "path":
        pattern = glob.escape(os.path.expanduser(buffer)) + "*"
        matches = []
        home = str(Path.home())
        for match in glob.glob(pattern):
            display = match
            if buffer.startswith("~"):
                try:
                    relative = os.path.relpath(match, home)
                    if relative == ".":
                        display = "~"
                    elif relative != ".." and not relative.startswith(".." + os.sep):
                        display = "~" + os.sep + relative
                except ValueError:
                    pass
            if Path(match).is_dir() and not display.endswith(os.sep):
                display += os.sep
            matches.append(display)
        return sorted(set(matches), key=str.casefold)
    if completion == "gpu":
        boundary = max(buffer.rfind(" "), buffer.rfind("\t")) + 1
        prefix, fragment = buffer[:boundary], buffer[boundary:]
        return [prefix + str(index) for index in range(16) if str(index).startswith(fragment)]
    if completion is None:
        return []
    folded = buffer.casefold()
    return [candidate for candidate in completion if candidate.casefold().startswith(folded)]


def _readline_input(prompt: str, completion: Sequence[str] | str | None) -> str:
    import readline

    previous_completer = readline.get_completer()
    previous_delimiters = readline.get_completer_delims()
    previous_append = (readline.get_completion_append_character()
                       if hasattr(readline, "get_completion_append_character") else None)
    candidates: list[str] = []
    last_buffer = ""

    def complete(text: str, state: int) -> str | None:
        nonlocal candidates, last_buffer
        buffer = readline.get_line_buffer()
        if state == 0 or buffer != last_buffer:
            last_buffer = buffer
            candidates = _completion_matches(buffer, completion)
            if not candidates:
                sys.stdout.write("\nNo matches for this field.\n")
                sys.stdout.flush()
        return candidates[state] if state < len(candidates) else None

    try:
        readline.set_completer(complete)
        readline.set_completer_delims("\t\n")
        readline.parse_and_bind("tab: complete")
        readline.parse_and_bind('"\\C-f": complete')
        if previous_append is not None and hasattr(readline, "set_completion_append_character"):
            try:
                readline.set_completion_append_character("")
            except ValueError:
                pass
        return input(prompt)
    finally:
        readline.set_completer(previous_completer)
        readline.set_completer_delims(previous_delimiters)
        if previous_append is not None and hasattr(readline, "set_completion_append_character"):
            readline.set_completion_append_character(previous_append)


def _raw_terminal_input(prompt: str, completion: Sequence[str] | str | None) -> str:
    """Line editor for interactive terminals without Python's readline module."""
    if os.name == "nt":
        import msvcrt

        def read_key() -> str:
            key = msvcrt.getwch()
            if key in ("\x00", "\xe0"):
                msvcrt.getwch()
                return ""
            return key
    else:
        import termios
        import tty

        file_descriptor = sys.stdin.fileno()
        previous_mode = termios.tcgetattr(file_descriptor)
        tty.setraw(file_descriptor)

        def read_key() -> str:
            key = sys.stdin.read(1)
            if key == "\x1b":
                next_key = sys.stdin.read(1)
                if next_key == "[":
                    sys.stdin.read(1)
                return ""
            return key

    buffer: list[str] = []
    sys.stdout.write(prompt)
    sys.stdout.flush()
    try:
        while True:
            key = read_key()
            if key in ("\r", "\n"):
                sys.stdout.write("\r\n")
                return "".join(buffer)
            if key in ("\x03", "\x04", "\x1a"):
                if key == "\x03":
                    raise KeyboardInterrupt
                if buffer:
                    continue
                raise EOFError
            if key in ("\b", "\x7f"):
                if buffer:
                    buffer.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue
            if key == "\x15":
                sys.stdout.write("\b \b" * len(buffer))
                buffer.clear()
                sys.stdout.flush()
                continue
            if key in ("\t", "\x06"):
                current = "".join(buffer)
                matches = _completion_matches(current, completion)
                layout = _layout()
                if not matches:
                    sys.stdout.write("\r\n")
                    for line in wrap_cells("No matches for this field.", layout.width):
                        sys.stdout.write(layout.indent + line + "\r\n")
                else:
                    common = os.path.commonprefix(matches)
                    if common.startswith(current) and len(common) > len(current):
                        suffix = common[len(current):]
                        buffer.extend(suffix)
                        sys.stdout.write(suffix)
                    if len(matches) > 1:
                        sys.stdout.write("\r\n")
                        for line in wrap_cells("\n".join(matches[:40]), layout.width, literal=True):
                            sys.stdout.write(layout.indent + line + "\r\n")
                        if len(matches) > 40:
                            count = _fit(f"… {len(matches) - 40} more", layout.width)
                            sys.stdout.write(layout.indent + count + "\r\n")
                sys.stdout.write(prompt + "".join(buffer))
                sys.stdout.flush()
                continue
            if key and key.isprintable():
                buffer.append(key)
                sys.stdout.write(key)
                sys.stdout.flush()
    finally:
        if os.name != "nt":
            termios.tcsetattr(file_descriptor, termios.TCSADRAIN, previous_mode)


def _read_input(prompt: str, completion: Sequence[str] | str | None = None) -> str:
    if not sys.stdin.isatty():
        print(prompt.rstrip(), flush=True)
        return input()
    try:
        import readline  # noqa: F401
    except ImportError:
        return _raw_terminal_input(prompt, completion)
    return _readline_input(prompt, completion)


def _ask(label: str, default: str = "", *, detail: str = "", technical: str = "",
         completion: Sequence[str] | str | None = None) -> str:
    print()
    _write(label, "heading")
    if detail:
        _write(detail, "muted")
    if VERBOSE and technical:
        _write(technical, "accent")
    suffix = f" [{default}]" if default else ""
    if default and cell_width(suffix) > _layout().width - 12:
        _write(f"Default  {default}", "muted", literal=True)
        suffix = ""
    # Readline needs a plain prompt to calculate cursor and wrapping positions.
    answer = _read_input(f"{_layout().indent}›{suffix}: ", completion).strip()
    return answer or default


def _ask_choice(label: str, choices: Sequence[str], default: str, *, detail: str = "", technical: str = "") -> str:
    while True:
        answer = _ask(label, default, detail=detail, technical=technical,
                      completion=choices).casefold()
        if answer in choices:
            return answer
        _write("Choose one of: " + ", ".join(choices), "error")


def _ask_nonnegative(label: str, default: str, *, allow_auto: bool = True, integer: bool = False,
                     detail: str = "", technical: str = "") -> str:
    while True:
        examples = (["auto"] if allow_auto else []) + ["0", default]
        answer = _ask(label, default, detail=detail, technical=technical,
                      completion=list(dict.fromkeys(examples)))
        if allow_auto and answer.casefold() == "auto":
            return "auto"
        try:
            number = int(answer) if integer else float(answer)
        except (OverflowError, ValueError):
            number = math.nan
        if number >= 0 and (integer or math.isfinite(number)):
            return answer
        _write("Enter a finite number greater than or equal to zero" +
               (", or auto." if allow_auto else "."), "error")


def _ask_integer(label: str, default: str, minimum: int = 0, *,
                 detail: str = "", technical: str = "") -> str:
    while True:
        answer = _ask(label, default, detail=detail, technical=technical,
                      completion=list(dict.fromkeys([str(minimum), default])))
        try:
            number = int(answer)
        except ValueError:
            number = minimum - 1
        if number >= minimum:
            return str(number)
        _write(f"Enter an integer greater than or equal to {minimum}.", "error")


def _ask_path(label: str, default: str = "", *, detail: str = "", technical: str = "") -> Path:
    completion_hint = "Tab or Ctrl+F completes matching paths."
    detail = f"{detail}  {completion_hint}" if detail else completion_hint
    while True:
        value = _ask(label, default, detail=detail, technical=technical, completion="path")
        if value:
            return _path(value)
        _write("Please enter a path.", "error")


def _confirm(label: str = "Start this workflow?", *, detail: str = "", technical: str = "") -> bool:
    return _ask(label + " [y/N]", detail=detail, technical=technical,
                completion=["y", "yes", "n", "no"]).casefold() in {"y", "yes"}


def _shell_command(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def _preview(command: Sequence[str], *, root: Path | None = None, cwd: Path = PROJECT_ROOT,
             summary: Sequence[str] = ()) -> None:
    _panel([
        "✦  READY TO RUN  ✦",
        f"Working directory  {cwd}",
        f"ZOO_DATA           {root if root else '(inherited if set)'}",
        *summary,
        "",
        _shell_command(command),
    ], literal=True)


def _run(command: Sequence[str], *, root: Path | None = None, cwd: Path = PROJECT_ROOT,
         summary: Sequence[str] = ()) -> int | None:
    _preview(command, root=root, cwd=cwd, summary=summary)
    if not _confirm():
        _write("Cancelled. No command was started.", "muted")
        return None
    environment = os.environ.copy()
    if root is not None:
        environment["ZOO_DATA"] = str(root)
    progress = Progress("Workflow", 1, lambda line: _write(line, "brand", flush=True))
    # Collector commands report their own row and worker progress.
    reports_progress = any(Path(arg).name == "run_experiment.py" for arg in command[:2])
    if not reports_progress:
        progress.update(0)
    print()
    try:
        outcome = subprocess.run(list(command), cwd=cwd, env=environment, check=False)
    except FileNotFoundError as error:
        _write(f"Could not start command: {error}", "error")
        return 127
    if not reports_progress:
        progress.update(1)
    print()
    color = "brand" if outcome.returncode == 0 else "error"
    _write(f"Finished with exit code {outcome.returncode}.", color)
    return outcome.returncode


def _data_root(current: Path | None) -> Path:
    default = str(current) if current else os.environ.get("ZOO_DATA", "")
    while True:
        value = _ask("Zoo data directory", default,
                     detail="Enter an existing artifact directory; Tab completes local paths.",
                     completion="path")
        if not value:
            _write("Set ZOO_DATA or enter a data directory to continue.", "error")
            continue
        root = _path(value)
        if not root.is_dir():
            _write(f"Directory not found: {root}", "error")
            continue
        return root


def _status(root: Path | None, *, compact: bool = False) -> list[str]:
    run_root = PROJECT_ROOT / "analysis_runs" / "phase2"
    run_count = sum(1 for item in run_root.iterdir() if item.is_dir()) if run_root.is_dir() else 0
    if root is None:
        root_line = "DATA ROOT   not configured · set ZOO_DATA to connect your workspace"
        package_count = 0
        snapshot_state = "not built"
        model_count = None
    else:
        package_dir = root / "packages"
        package_count = sum(1 for _ in package_dir.rglob("package.json")) if package_dir.is_dir() else 0
        snapshot = root / "builds" / "current"
        snapshot_state = "ready" if snapshot.exists() else "not built"
        model_count = None
        receipt = snapshot / "collection.json"
        if receipt.is_file():
            try:
                model_count = len(json.loads(receipt.read_text(encoding="utf-8")).get("models", []))
            except (OSError, ValueError, TypeError):
                pass
        root_line = f"DATA ROOT   {root}"
    model_label = f"{model_count:,} models" if model_count is not None else snapshot_state
    if compact:
        return [root_line, f"PACKAGES {package_count:,}  ·  ZOO {model_label}  ·  RUNS {run_count:,}"]
    return [
        root_line,
        f"PACKAGES    {package_count:,} published packages",
        f"ZOO BUILD   {model_label} · {snapshot_state if model_count is not None else ''}".rstrip(" ·"),
        f"RUNS        {run_count:,} folders in analysis_runs/phase2",
    ]


def _existing_parent(path: Path) -> Path:
    candidate = path.expanduser()
    while not candidate.exists() and candidate.parent != candidate:
        candidate = candidate.parent
    return candidate


def _disk_line(label: str, path: Path) -> str:
    try:
        usage = shutil.disk_usage(_existing_parent(path))
    except OSError:
        return f"{label:<13} unavailable · {path}"
    free_gib = usage.free / 1024**3
    total_gib = usage.total / 1024**3
    return f"{label:<13} {free_gib:.1f} GiB free / {total_gib:.1f} GiB · {path}"


def _preflight(gpu_ids: Sequence[int], root: Path | None, cache_root: Path) -> list[str]:
    """Read-only local host check; this launcher does not create cloud instances."""
    lines = [
        f"HOST          {platform.node()} · {platform.system()} {platform.machine()}",
        f"CPU           {os.cpu_count() or 'unknown'} logical cores",
    ]
    try:
        import psutil
        memory = psutil.virtual_memory()
        lines.append(f"HOST MEMORY   {memory.available / 1024**3:.1f} GiB available / {memory.total / 1024**3:.1f} GiB")
    except (ImportError, OSError):
        lines.append("HOST MEMORY   unavailable (psutil not installed)")

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        lines.append("GPU CHECK     nvidia-smi not found on PATH")
    else:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            if result.returncode != 0:
                lines.append("GPU CHECK     nvidia-smi could not read GPU status")
            else:
                devices = {}
                for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
                    if len(row) >= 5:
                        try:
                            devices[int(row[0])] = row
                        except ValueError:
                            continue
                for gpu_id in gpu_ids:
                    row = devices.get(gpu_id)
                    if row is None:
                        lines.append(f"GPU {gpu_id:<8} not reported by nvidia-smi")
                    else:
                        total, free, utilization = float(row[2]), float(row[3]), row[4]
                        lines.append(f"GPU {gpu_id:<8} {row[1]} · {free / 1024:.1f} / {total / 1024:.1f} GiB free · {utilization}% busy")
        except (OSError, subprocess.TimeoutExpired, ValueError):
            lines.append("GPU CHECK     could not complete within 10 seconds")

    if root is not None:
        if root.is_dir():
            lines.append(_disk_line("ZOO STORAGE", root))
        else:
            lines.append(f"ZOO STORAGE   directory not found · {root}")
    cache_parent = _existing_parent(cache_root)
    cache_state = "ready" if cache_root.is_dir() else "will create"
    cache_access = "writable parent" if os.access(cache_parent, os.W_OK) else "parent not writable"
    lines.append(_disk_line("WORKER CACHE", cache_root) + f" · {cache_state} · {cache_access}")
    token_configured = "HF_TOKEN" in os.environ or "HUGGING_FACE_HUB_TOKEN" in os.environ
    lines.append("HUB ACCESS    token environment variable set (value hidden)" if token_configured
                 else "HUB ACCESS    no token environment variable · public models can still work")
    lines.append("COMPUTE       uses GPUs already attached to this host; no cloud provisioning")
    return lines


def _animated_preflight(gpu_ids: Sequence[int], root: Path | None, cache_root: Path) -> list[str]:
    if not sys.stdout.isatty() or not MOTION or not _supports_full_screen():
        return _preflight(gpu_ids, root, cache_root)
    complete = threading.Event()
    outcome: dict[str, object] = {}

    def scan() -> None:
        try:
            outcome["lines"] = _preflight(gpu_ids, root, cache_root)
        except Exception as error:
            outcome["error"] = error
        finally:
            complete.set()

    worker = threading.Thread(target=scan, name="model-zoo-preflight", daemon=True)
    worker.start()
    frames = ("·", "✧", "·", "✧")
    frame_index = 0
    while not complete.is_set():
        index = frame_index % len(frames)
        frame = frames[index]
        dots = "." * (frame_index % 4)
        layout = _layout()
        status = _fit(f"{frame} Reading local GPU, memory, and storage status{dots}", layout.width)
        sys.stdout.write("\r\x1b[K" + layout.indent + THEME.paint(status, "accent"))
        sys.stdout.flush()
        complete.wait(0.12)
        frame_index += 1
    worker.join()
    sys.stdout.write("\r\x1b[K")
    sys.stdout.flush()
    if "error" in outcome:
        raise outcome["error"]  # type: ignore[misc]
    return outcome["lines"]  # type: ignore[return-value]


def _prepare(root: Path | None) -> None:
    models = _ask_path("Input CSV", "models.csv",
                       detail="One Hugging Face repository per row; the required column is model_id.",
                       technical="Preparation writes pinned revisions and ready/held reasons to <run>/models.csv.")
    if not models.is_file():
        _write(f"Input CSV not found: {models}", "error")
        return
    default_run = PROJECT_ROOT / "analysis_runs" / "phase2" / "my_run"
    run_dir = _ask_path("Run output folder", str(default_run),
                        detail="Choose a new folder for a fresh pin review, or a prepared folder to resume its existing list.",
                        technical="An existing models.csv is preserved; prepare a new folder to change the pinned input list.")
    limit = _ask("Optional row limit", detail="Leave blank to prepare every row in the input CSV.")
    command = [sys.executable, str(PROJECT_ROOT / "esd_experiment" / "run_experiment.py"),
               "--model_list", str(models), "--output_dir", str(run_dir), "--prepare_only"]
    if limit:
        command.extend(["--limit", limit])
    _run(command, root=root)


def _gpu_ids() -> list[int]:
    while True:
        gpu_text = _ask("GPU IDs to use (space separated)", "0",
                        detail="Example: 0 1 2 3. Choose devices already allocated to this host or shared server.",
                        technical="--gpus; the dispatcher only reserves devices named here.",
                        completion="gpu")
        try:
            gpu_ids = [int(value) for value in gpu_text.split()]
        except ValueError:
            gpu_ids = []
        if gpu_ids and all(value >= 0 for value in gpu_ids) and len(set(gpu_ids)) == len(gpu_ids):
            return gpu_ids
        _write("Enter unique, nonnegative GPU IDs such as 0 1 2 3.", "error")


def _cache_default() -> Path:
    configured = os.environ.get("MODEL_ZOO_WORKER_CACHE_ROOT")
    path = Path(configured).expanduser() if configured else Path(
        f"/scratch/{os.environ.get('USER') or getpass.getuser()}/hf_worker_cache")
    return path.resolve()


def _ask_stage_timeouts() -> str:
    while True:
        answer = _ask(
            "Stage timeouts", "auto",
            detail="Use auto, or enter comma-separated stage=seconds values; 0 disables a stage timeout.",
            technical="--stage_timeout_seconds; common stages are load, analyze, fingerprint, save, and default.",
            completion=["auto", "load=7200,analyze=28800,save=1800"],
        )
        if answer.casefold() == "auto":
            return "auto"
        entries = [entry.strip() for entry in answer.split(",")]
        try:
            valid = bool(entries) and all(
                "=" in entry and bool(entry.split("=", 1)[0].strip())
                and int(entry.split("=", 1)[1].strip()) >= 0
                for entry in entries
            )
        except ValueError:
            valid = False
        if valid:
            return answer
        _write("Use auto or comma-separated stage=nonnegative-seconds values.", "error")


def _infrastructure(root: Path | None = None) -> None:
    gpu_ids = _gpu_ids()
    cache_root = _ask_path("Worker cache location", str(_cache_default()))
    _panel(_animated_preflight(gpu_ids, root, cache_root))


def _launch(root: Path | None) -> None:
    run_dir = _ask_path("Prepared run folder", str(PROJECT_ROOT / "analysis_runs" / "phase2" / "my_run"),
                        detail="This folder should contain the pinned models.csv from preparation.")
    model_list = run_dir / "models.csv"
    if not model_list.is_file():
        _write(f"No prepared model list found: {model_list}", "error")
        return

    profile = _ask_choice(
        "Resource plan: 1 guided / 2 conservative quick start", ["1", "2"], "1",
        detail="Guided exposes each resource envelope. Quick start keeps one worker and automatic caps.",
        technical="Quick start still asks about GPUs, cache placement, hub code, and host preflight.",
    )
    gpu_ids = _gpu_ids()
    cache_root = _ask_path("Ephemeral worker cache root", str(_cache_default()),
                           detail="Model downloads and worker state go here while each job runs; free space matters.",
                           technical="This runner owns and removes per-worker caches after confirmed cleanup; shared caches stay untouched.")

    options = {
        "--num_gpus_per_job": "auto",
        "--max_concurrent_jobs": "1",
        "--max_weight_gib": "auto",
        "--max_worker_ram_gib": "auto",
        "--max_worker_cache_gib": "auto",
        "--cpu_threads_per_gpu": "auto",
        "--prefetch_models": "auto",
        "--prefetch_max_gib": "64",
    }
    if profile == "1":
        options["--num_gpus_per_job"] = _ask_choice(
            "GPUs assigned per model", ["auto"] + [str(number) for number in range(1, len(gpu_ids) + 1)], "auto",
            detail="Auto uses reviewed weight size and the smallest selected GPU’s total memory.",
            technical="Automatic sizing is a heuristic; an assigned GPU does not guarantee enough analysis memory.",
        )
        options["--max_concurrent_jobs"] = _ask_integer(
            "Maximum model workers at once", "1", minimum=1,
            detail="More workers can improve throughput when GPU, RAM, and disk capacity allow it.",
            technical="This is a worker count, not a host-wide memory limit. Per-worker caps multiply across active jobs.",
        )
        options["--max_weight_gib"] = _ask_nonnegative(
            "Maximum selected model weight size in GiB", "auto",
            detail="Auto reviews model and adapter-base weight sizes against the selected GPU pool.",
            technical="0 removes the size cap; missing-file, size, and access checks still apply.",
        )
        options["--max_worker_ram_gib"] = _ask_nonnegative(
            "Worker-tree RAM cap in GiB", "auto",
            detail="Auto chooses a per-worker cap based on reviewed weight size.",
            technical="The supervisor samples process-tree RSS every five seconds; this is a safety stop, not an OS quota. 0 disables it.",
        )
        options["--max_worker_cache_gib"] = _ask_nonnegative(
            "Private worker-cache cap in GiB", "auto",
            detail="Auto scales with selected weight size; use the preflight free-space reading when choosing a cap.",
            technical="This is per worker and sampled; fast writes can overshoot before the next check. 0 disables the cap.",
        )
        options["--cpu_threads_per_gpu"] = _ask_nonnegative(
            "CPU threads per assigned GPU", "auto", integer=True,
            detail="Auto divides host CPU cores across the physical GPU pool.",
            technical="0 preserves inherited thread settings; a positive value sets limits for OMP, MKL, OpenBLAS, and NumExpr.",
        )
        options["--prefetch_models"] = _ask_nonnegative(
            "Models to download ahead", "auto", integer=True,
            detail="Auto bounds download-ahead by the GPU pool; choose 0 to disable prefetch.",
            technical="Prefetch uses future workers’ private caches and falls back to worker downloads when it fails.",
        )
        options["--prefetch_max_gib"] = _ask_nonnegative(
            "Total download-ahead budget in GiB", "64", allow_auto=False,
            detail="Caps staged downloads while workers are running.",
            technical="This prefetch budget is separate from each active worker’s private-cache cap.",
        )

    advanced = _confirm(
        "Tune GPU admission and worker recovery safeguards?",
        detail="Defaults: two idle checks, a 500 MiB admission threshold, and supervised termination.",
        technical="Advanced values map to the run dispatcher and worker-supervisor flags.",
    )
    if advanced:
        threshold = _ask_integer(
            "GPU used-memory admission threshold in MiB", "500", minimum=0,
            detail="A GPU is considered idle only below this used-memory threshold and with no detected GPU processes.",
            technical="--gpu_memory_threshold; another user can still start a job after an idle check.",
        )
        checks = _ask_integer(
            "Consecutive idle checks before dispatch", "2", minimum=1,
            detail="Checks are five seconds apart; more checks reduce races but slow admission.",
            technical="--max_check; two checks are the documented default.",
        )
        options["--stale_process_action"] = _ask_choice(
            "Expired worker timeout action", ["terminate", "log"], "terminate",
            detail="Terminate requests supervised cleanup; log records the stale state and keeps waiting.",
            technical="--stale_process_action; termination only releases devices after worker exit is confirmed.",
        )
        options["--heartbeat_timeout_seconds"] = _ask_integer(
            "Heartbeat timeout in seconds", "3600", minimum=0,
            detail="Time without a worker heartbeat before the supervisor marks it stale. 0 disables this timeout.",
            technical="--heartbeat_timeout_seconds; stage timeouts independently catch stuck foreground work.",
        )
        options["--stage_timeout_seconds"] = _ask_stage_timeouts()
        options["--termination_grace_seconds"] = _ask_integer(
            "Grace period after TERM before KILL", "30", minimum=1,
            detail="Gives a stale worker group time to shut down cleanly.",
            technical="--termination_grace_seconds; cache cleanup and GPU release follow confirmed exit.",
        )
    else:
        threshold, checks = "500", "2"

    _panel(_animated_preflight(gpu_ids, root, cache_root))
    command = [sys.executable, str(PROJECT_ROOT / "esd_experiment" / "run_experiment.py"),
               "--model_list", str(model_list), "--output_dir", str(run_dir),
               "--gpus", *map(str, gpu_ids), "--worker_cache_root", str(cache_root),
               "--gpu_memory_threshold", threshold, "--max_check", checks]
    for flag, value in options.items():
        command.extend([flag, value])

    summary = [
        f"GPU pool: {', '.join(map(str, gpu_ids))} · {options['--num_gpus_per_job']} GPU(s) per model",
        f"Workers: {options['--max_concurrent_jobs']} · weight/RAM/cache budgets: "
        f"{options['--max_weight_gib']} / {options['--max_worker_ram_gib']} / {options['--max_worker_cache_gib']} GiB",
        f"CPU threads/GPU: {options['--cpu_threads_per_gpu']} · prefetch: "
        f"{options['--prefetch_models']} models / {options['--prefetch_max_gib']} GiB",
        f"GPU admission: under {threshold} MiB · {checks} idle checks · "
        f"stale action: {options.get('--stale_process_action', 'terminate')}",
        f"Recovery timeouts: heartbeat {options.get('--heartbeat_timeout_seconds', '3600')}s · "
        f"stages {options.get('--stage_timeout_seconds', 'auto')} · "
        f"TERM grace {options.get('--termination_grace_seconds', '30')}s",
        f"Worker cache: {cache_root}",
        "Remote repository code: disabled",
        "Resource limits are per worker and sampled; they are not hard host quotas.",
    ]
    _run(command, root=root, summary=summary)


def _analyze(root: Path | None) -> None:
    run_dir = _ask_path("Run folder", str(PROJECT_ROOT / "analysis_runs" / "phase2" / "my_run"),
                        detail="Reads run results and writes the rebuildable summary.csv index.")
    if not run_dir.is_dir():
        _write(f"Run folder not found: {run_dir}", "error")
        return
    command = [sys.executable, str(PROJECT_ROOT / "esd_experiment" / "analyze_results.py"),
               "--results_dir", str(run_dir)]
    _run(command, root=root)


def _publish(root: Path | None) -> None:
    run_text = _ask("Completed run folders",
                    detail="Use semicolons between paths; Tab completes the path currently being entered.",
                    completion="multi_path")
    runs = [_path(value.strip()) for value in run_text.split(";") if value.strip()]
    if not runs:
        _write("Enter at least one run folder.", "error")
        return
    missing = [run for run in runs if not run.is_dir()]
    if missing:
        _write("Run folder not found: " + ", ".join(map(str, missing)), "error")
        return
    selected_root = _data_root(root)
    command = [sys.executable, "-m", "zoo.cli", "publish", "--runs", *map(str, runs),
               "--data_root", str(selected_root)]
    _run(command, root=selected_root)


def _build(root: Path | None) -> None:
    selected_root = _data_root(root)
    command = [sys.executable, "-m", "zoo.cli", "build", "--data_root", str(selected_root)]
    _run(command, root=selected_root)


def _deltas(root: Path | None) -> None:
    selected_root = _data_root(root)
    workers = _ask("Workers", "auto", detail="Automatic worker sizing follows the detected candidate set.",
                   completion=["auto", "1", "2", "4"])
    device = _ask_choice("Compute device", ["auto", "cpu", "cuda"], "auto",
                         detail="Auto follows the delta command’s device selection policy.")
    command = [sys.executable, "-m", "zoo.cli", "deltas", "--workers", workers,
               "--compute_device", device, "--data_root", str(selected_root)]
    _run(command, root=selected_root)


def _query(root: Path | None) -> None:
    query_root = QUERY_ROOT if QUERY_ROOT.is_dir() else PROJECT_ROOT
    try:
        available = importlib.util.find_spec("zoo_keeper") is not None
    except (ImportError, ValueError):
        available = False
    available = available or (QUERY_ROOT / "zoo_keeper").is_dir()
    if not available:
        _panel([
            "The query interface is provided by the companion Zoo-Keeper project.",
            "In its checkout, install Model-Zoo and duckdb, then run:",
            "  python -m zoo_keeper.cli find --text gpt2 --limit 10",
            "  python -m zoo_keeper.cli similar openai-community/gpt2 -k 5",
        ])
        return
    selected_root = _data_root(root)
    zoo_dir = selected_root / "builds" / "current"
    _write("1  Find models by name or metadata\n2  Find models with similar weight fingerprints")
    choice = _ask_choice("Choose a query", ("1", "2"), "1")
    if choice == "1":
        term = _ask("Search text")
        if not term:
            return
        limit = _ask_integer("Maximum results", "10", minimum=1,
                             detail="Examples: 10, 25, 50.")
        command = [sys.executable, "-m", "zoo_keeper.cli", "find", "--text", term,
                   "--limit", limit, "--zoo", str(zoo_dir)]
    elif choice == "2":
        model = _ask("Model id (org/name or repo@revision)")
        if not model:
            return
        count = _ask_integer("Nearest models", "5", minimum=1,
                             detail="Examples: 5, 10, 25.")
        command = [sys.executable, "-m", "zoo_keeper.cli", "similar", model,
                   "-k", count, "--zoo", str(zoo_dir)]
    _run(command, root=selected_root, cwd=query_root)


def _pause() -> None:
    try:
        _read_input("\n" + _layout().indent + "Enter  Back to workflows")
    except EOFError:
        pass


def _display_width(text: str) -> int:
    return cell_width(text)


def _fit(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if _display_width(text) <= width:
        return text
    output, used = [], 0
    for character in text:
        character_width = 0 if unicodedata.combining(character) else (
            2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1)
        if used + character_width > width - 1:
            break
        output.append(character)
        used += character_width
    return "".join(output) + "…"


def _draw_home(root: Path | None) -> None:
    """Static, accessible fallback for terminals without curses support."""
    from zoo.terminal_home import WORKFLOWS

    _clear()
    _write("MODEL ZOO", "brand")
    for line in _status(root, compact=True):
        _write(line, "muted", wrap=False)
    print()
    _write("Workflows", "accent")
    for item in WORKFLOWS:
        _write(f"{item.key}  {item.title}", wrap=False)
    _write("Enter a number  ·  Q Quit", "muted", wrap=False)


def _read_menu_choice(root: Path | None) -> str:
    return _read_input(_layout().indent + "Workflow [1–8/q] › ")


def _supports_full_screen() -> bool:
    terminal = os.environ.get("TERM", "")
    if terminal:
        return terminal != "dumb"
    if os.name != "nt":
        return False
    try:
        import curses
    except ImportError:
        return False
    return True


@contextmanager
def _workflow_screen(choice: str, *, enabled: bool = True):
    """Keep guided prompts in a fresh screen and restore the shell on exit."""
    from zoo.terminal_home import WORKFLOWS
    interactive = enabled and sys.stdout.isatty() and _supports_full_screen()
    if interactive:
        THEME.in_frame = True
        sys.stdout.write("\x1b[?1049h" + THEME.backdrop() + "\x1b[2J\x1b[H")
        sys.stdout.flush()
    try:
        item = next(item for item in WORKFLOWS if item.key == choice)
        print()
        _write(f"MODEL ZOO  /  {item.group}", "brand", wrap=False)
        _write(item.title, "heading")
        _write(item.description, "muted")
        _write("─" * _layout().width, "border", wrap=False)
        print()
        yield
    finally:
        if interactive:
            if THEME.enabled:
                sys.stdout.write("\x1b[0m")
            THEME.in_frame = False
            sys.stdout.write("\x1b[?1049l")
            sys.stdout.flush()


def main(argv: Sequence[str] | None = None) -> int:
    global VERBOSE, MOTION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        help="Zoo artifact root (defaults to ZOO_DATA)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show technical notes and infrastructure details beside prompts")
    parser.add_argument("--theme", choices=("auto", "dark", "light"), default="auto",
                        help="Terminal background palette (default: detect background or use dark)")
    parser.add_argument("--no-motion", action="store_true", help="Disable ambient animation")
    parser.add_argument("--plain", action="store_true", help="Use numbered prompts without full-screen navigation")
    args = parser.parse_args(argv)
    VERBOSE = args.verbose
    THEME.palette = THEME.automatic_palette() if args.theme == "auto" else args.theme
    root = args.data_root.expanduser().resolve() if args.data_root else None
    if root is None and os.environ.get("ZOO_DATA"):
        root = Path(os.environ["ZOO_DATA"]).expanduser().resolve()
    actions = {
        "1": _prepare,
        "2": _launch,
        "3": _analyze,
        "4": _publish,
        "5": _build,
        "6": _deltas,
        "7": _query,
        "8": _infrastructure,
    }
    motion = not args.no_motion and "NO_COLOR" not in os.environ
    THEME.enabled = Theme().enabled
    full_screen = (not args.plain and sys.stdin.isatty() and sys.stdout.isatty()
                   and _supports_full_screen())
    while True:
        try:
            if full_screen:
                from zoo.terminal_home import navigate
                try:
                    import curses
                except ImportError:
                    full_screen = False
                    continue
                try:
                    choice, THEME.palette, motion = navigate(
                        _status(root, compact=True), palette=THEME.palette,
                        motion=motion, color=THEME.enabled)
                except curses.error:
                    full_screen = False
                    continue
            else:
                _draw_home(root)
                choice = _read_menu_choice(root)
            choice = choice.strip().casefold()
        except (EOFError, KeyboardInterrupt):
            print("\nSee you next time.")
            return 0
        if choice in {"q", "quit", "exit"}:
            print("See you next time.")
            return 0
        action = actions.get(choice)
        if action is None:
            _write("Choose a number from 1 to 8, or q to quit.", "error")
            _pause()
            continue
        MOTION = motion
        with _workflow_screen(choice, enabled=full_screen):
            try:
                action(root)
            except (KeyboardInterrupt, EOFError):
                _write("\nWorkflow cancelled.", "muted")
            except (OSError, ValueError) as error:
                _write(f"Could not prepare workflow: {error}", "error")
            _pause()



if __name__ == "__main__":
    raise SystemExit(main())
