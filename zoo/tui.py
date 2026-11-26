"""A guided terminal launcher for Model-Zoo's collection workflows."""

from __future__ import annotations

import argparse
from collections import deque
import codecs
import csv
from dataclasses import replace
import glob
import importlib.util
from itertools import groupby
import os
from pathlib import Path
import platform
from secrets import randbits
import shlex
import shutil
import subprocess
import sys
import time
import unicodedata
from typing import NamedTuple, Sequence

from zoo.terminal_style import PageLayout, Theme, cell_width, wrap_cells
from zoo.workspace import Workspace
from zoo.terminal_home import ANIMATION_FPS, MotionClock, brand_header, branded_footer, star_color, star_rgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUERY_ROOT = PROJECT_ROOT.parent / "Zoo-Keeper"
VERBOSE = False
THEME = Theme()
STAR_SEED = randbits(64)
CLOCK = MotionClock()


class TerminalPage(NamedTuple):
    size: tuple[int, int]
    reserve_rows: int
    header_rows: int
    header_budget: int
    footer_hint: str
    stars: dict[tuple[int, int], str]
    body: deque[tuple[str, str, bool, bool]]


_page: TerminalPage | None = None


def _layout() -> PageLayout:
    return PageLayout(shutil.get_terminal_size((84, 24)).columns)


def _formatted_lines(text: str, role: str, wrap: bool, literal: bool) -> list[str]:
    layout = _layout()
    lines = wrap_cells(text, layout.width, literal=literal) if wrap else [_fit(text, layout.width)]
    return [layout.indent + THEME.paint(line, role) for line in lines]


def _write(text: str, role: str = "text", *, wrap: bool = True, literal: bool = False,
           flush: bool = False) -> None:
    entry = (text, role, wrap, literal)
    if _page is not None:
        _page.body.append(entry)
    for line in _formatted_lines(*entry):
        print(line, flush=flush)


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _completion_matches(buffer: str, completion: Sequence[str] | str | None) -> list[str]:
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
    if completion is None:
        return []
    folded = buffer.casefold()
    return [candidate for candidate in completion if candidate.casefold().startswith(folded)]


class Back(Exception):
    """Return to the previous launcher screen."""


def _posix_key(fd: int, decoder) -> str:
    from select import select

    data = os.read(fd, 1)
    if not data:
        raise EOFError
    if data != b"\x1b":
        return decoder.decode(data)

    # Escape on its own goes back. Arrow and function keys send a sequence.
    def next_byte() -> bytes:
        return os.read(fd, 1) if select([fd], [], [], 0.05)[0] else b""

    prefix = next_byte()
    if not prefix or prefix == b"\x1b":
        return "\x1b"
    if prefix not in (b"[", b"O"):
        return ""  # Alt plus a key
    sequence = b""
    while byte := next_byte():
        sequence += byte
        if b"@" <= byte <= b"~":
            break
    return {
        b"D": "left", b"C": "right", b"H": "home", b"F": "end",
        b"1~": "home", b"4~": "end", b"3~": "delete", b"Z": "\x06",
    }.get(sequence, "")


def _raw_terminal_input(prompt: str, completion: Sequence[str] | str | None) -> str:
    """Keep editing, completion and Escape behavior consistent across consoles."""
    if os.name == "nt":
        import msvcrt

        pending = ""

        def read_key() -> str:
            nonlocal pending
            key, pending = pending or msvcrt.getwch(), ""
            if key in ("\x00", "\xe0"):
                return {"K": "left", "M": "right", "G": "home", "O": "end", "S": "delete"}.get(
                    msvcrt.getwch(), "")
            if "\ud800" <= key <= "\udbff":
                low = msvcrt.getwch()
                if "\udc00" <= low <= "\udfff":
                    return (key + low).encode("utf-16-le", "surrogatepass").decode("utf-16-le")
                pending = low
                return "\ufffd"
            return key
    else:
        import termios
        import tty

        file_descriptor = sys.stdin.fileno()
        previous_mode = termios.tcgetattr(file_descriptor)
        decoder = codecs.getincrementaldecoder(sys.stdin.encoding or "utf-8")(errors="replace")
        tty.setraw(file_descriptor, termios.TCSANOW)

        def read_key() -> str:
            return _posix_key(file_descriptor, decoder)

    buffer = ""
    cursor = 0
    first_visible = 0
    shown = ""
    shown_cursor = 0

    def redraw() -> None:
        nonlocal first_visible, shown, shown_cursor
        # Keep the editable text on one row so Backspace works in both VT
        # terminals and the legacy Windows console, even for very long paths.
        first_visible = min(first_visible, cursor)
        column = cell_width(buffer[first_visible:cursor])
        while column > input_width:
            column -= cell_width(buffer[first_visible])
            first_visible += 1

        end = cursor
        remaining = input_width - column
        while end < len(buffer) and cell_width(buffer[end]) <= remaining:
            remaining -= cell_width(buffer[end])
            end += 1
        visible = buffer[first_visible:end]
        if visible != shown or column != shown_cursor:
            padding = " " * max(0, cell_width(shown) - cell_width(visible))
            sys.stdout.write("\b" * shown_cursor + visible + padding +
                             "\b" * (cell_width(visible + padding) - column))
            shown, shown_cursor = visible, column

    def edit(start: int, stop: int, replacement: str = "") -> None:
        nonlocal buffer, cursor
        buffer = buffer[:start] + replacement + buffer[stop:]
        cursor = start + len(replacement)

    try:
        prompt_text = prompt.lstrip()

        def fit_prompt() -> tuple[str, int]:
            columns = shutil.get_terminal_size((84, 24)).columns
            fitted = _fit(PageLayout(columns).indent + prompt_text, columns - 2)
            return fitted, max(0, columns - 1 - cell_width(fitted))

        leading_newline = prompt.startswith("\n")
        prompt, input_width = fit_prompt()
        sys.stdout.write(("\r\n" if leading_newline else "") + prompt)
        sys.stdout.flush()
        while True:
            if _page is not None:
                if shutil.get_terminal_size((84, 24)) != _page.size:
                    _resize_page()
                    prompt, input_width = fit_prompt()
                    first_visible, shown, shown_cursor = 0, "", 0
                    sys.stdout.write(prompt)
                    redraw()
                    sys.stdout.flush()
                elif CLOCK.started is not None:
                    _refresh_header()
                if os.name == "nt":
                    if not pending and not msvcrt.kbhit():
                        time.sleep(1 / ANIMATION_FPS)
                        continue
                else:
                    from select import select
                    if not select([file_descriptor], [], [], 1 / ANIMATION_FPS)[0]:
                        continue
            key = read_key()
            if key in ("\r", "\n"):
                sys.stdout.write("\r\n")
                return buffer
            if key == "\x1b":
                sys.stdout.write("\r\n")
                raise Back
            if key == "\x03":
                raise KeyboardInterrupt
            if key in ("\x04", "\x1a") and not buffer:
                raise EOFError
            if key in ("left", "home"):
                cursor = 0 if key == "home" else max(0, cursor - 1)
            elif key in ("right", "end"):
                cursor = len(buffer) if key == "end" else min(len(buffer), cursor + 1)
            elif key in ("\b", "\x7f") and cursor:
                edit(cursor - 1, cursor)
            elif key == "delete":
                edit(cursor, cursor + 1)
            elif key == "\x15":
                edit(0, len(buffer))
            elif key in ("\t", "\x06"):
                matches = _completion_matches(buffer, completion)
                prefix = os.path.commonprefix(matches)
                if prefix.startswith(buffer) and len(prefix) > len(buffer):
                    edit(cursor, len(buffer), prefix[cursor:])
                if len(matches) != 1:
                    sys.stdout.write("\r\n")
                    rows = matches[:40] if matches else ["No matches for this field."]
                    if len(matches) > 40:
                        rows.append(f"… {len(matches) - 40} more")
                    layout = _layout()
                    for line in wrap_cells("\n".join(rows), layout.width, literal=True):
                        sys.stdout.write(layout.indent + line + "\r\n")
                    sys.stdout.write(prompt)
                    shown, shown_cursor = "", 0
            elif len(key) == 1 and key.isprintable():
                edit(cursor, cursor, key)
            redraw()
            sys.stdout.flush()
    finally:
        if os.name != "nt":
            termios.tcsetattr(file_descriptor, termios.TCSADRAIN, previous_mode)
        sys.stdout.flush()


def _read_input(prompt: str, completion: Sequence[str] | str | None = None) -> str:
    if sys.stdin.isatty() and sys.stdout.isatty():
        return _raw_terminal_input(prompt, completion)
    print(prompt.rstrip(), flush=True)
    return input()


def _ask(label: str, default: str = "", *, detail: str = "",
         completion: Sequence[str] | str | None = None) -> str:
    print()
    _write(label, "heading")
    if detail:
        _write(detail, "muted")
    suffix = f" [{default}]" if default else ""
    if default and cell_width(suffix) > _layout().width - 12:
        _write(f"Default  {default}", "muted", literal=True)
        suffix = ""
    answer = _read_input(f"{_layout().indent}›{suffix}: ", completion).strip()
    if answer == "\x1b":
        raise Back
    return answer or default


def _ask_choice(label: str, choices: Sequence[str], default: str, *, detail: str = "") -> str:
    while True:
        answer = _ask(label, default, detail=detail,
                      completion=choices).casefold()
        if answer in choices:
            return answer
        _write("Choose " + " or ".join(choices) + ".", "error")


def _ask_path(label: str, default: str = "", *, detail: str = "") -> Path:
    completion_hint = "Tab or Ctrl+F completes matching paths."
    detail = f"{detail}  {completion_hint}" if detail else completion_hint
    while True:
        value = _ask(label, default, detail=detail, completion="path")
        if value:
            return _path(value)
        _write("Please enter a path.", "error")


def _confirm(label: str = "Start this workflow?", *, detail: str = "") -> bool:
    return _ask(label + " [y/N]", detail=detail,
                completion=["y", "yes", "n", "no"]).casefold() in {"y", "yes"}


def _shell_command(command: Sequence[str]) -> str:
    if os.name == "nt":
        return "& " + " ".join("'" + value.replace("'", "''") + "'" for value in command)
    return shlex.join(command)


def _run(command: Sequence[str], workspace: Workspace, *, label: str,
         cwd: Path = PROJECT_ROOT, summary: Sequence[str] = (), confirm: bool = False) -> int:
    if confirm:
        _clear_screen()
        _draw_brand()
    _write(label, "heading")
    if confirm and _page is None:
        _write(branded_footer(_layout().width, "Esc Back"), "muted", wrap=False)
    for line in summary:
        _write(line, literal=True)
    if VERBOSE:
        _write(f"Workspace (ZOO_DATA)  {workspace.root}", "muted", literal=True)
        _write(f"Process directory  {cwd}", "muted", literal=True)
        _write(_shell_command(command), "muted", literal=True)
    if confirm and not _confirm():
        _write("Cancelled.", "muted")
        return 0
    environment = dict(os.environ, ZOO_DATA=str(workspace.root), PYTHONUNBUFFERED="1")
    _write("Starting…", "muted", flush=True)
    _release_header()  # Let long command output use the terminal’s normal scrollback.
    outcome = subprocess.run(list(command), cwd=cwd, env=environment, check=False)
    if outcome.returncode == 0:
        _write("Done.", "success")
    else:
        _write(f"Workflow exited with code {outcome.returncode}. See the output above.", "error")
    return outcome.returncode


def _status(root: Path | None) -> list[str]:
    if root is None:
        return ["Workspace is unset"]
    snapshot = "ready to search" if (root / "builds" / "current" / "models.parquet").is_file() else "not built yet"
    return [f"Workspace  {root}", f"Catalog  {snapshot}"]


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


class Gpu(NamedTuple):
    name: str
    uuid: str
    total_mib: float
    free_mib: float
    busy_percent: str


def _gpu_inventory() -> dict[int, Gpu]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {}
    try:
        result = subprocess.run(
            [executable, "--query-gpu=index,name,uuid,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    devices = {}
    if result.returncode == 0:
        for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
            try:
                index, name, uuid, total, free, busy = row
                devices[int(index)] = Gpu(name, uuid, float(total), float(free), busy)
            except ValueError:
                continue
    return devices


def _gpu_choices(devices: dict[int, Gpu]) -> list[int]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return sorted(devices)
    permitted = []
    for value in visible.split(","):
        value = value.strip()
        if value.isascii() and value.isdecimal():
            index = int(value)
            if not devices or index in devices:
                permitted.append(index)
                continue
        elif value.startswith("GPU-"):
            matching = [index for index, gpu in devices.items() if gpu.uuid.startswith(value)]
            if len(matching) == 1:
                permitted.extend(matching)
                continue
        # A disabled or unknown entry ends the usable part of the CUDA list.
        break
    return list(dict.fromkeys(permitted))


def _gpu_line(index: int, device: Gpu) -> str:
    return (f"GPU {index}  {device.name} · {device.free_mib / 1024:.1f}/"
            f"{device.total_mib / 1024:.1f} GiB free · {device.busy_percent}% busy")


def _infrastructure(workspace: Workspace) -> int:
    _write("Checking this computer’s GPUs, memory and storage…", flush=True)
    devices = _gpu_inventory()
    visible = [index for index in _gpu_choices(devices) if index in devices]
    lines = [
        f"{platform.node()} · {platform.system()} · {os.cpu_count() or 'unknown'} logical CPU cores",
        *(_gpu_line(index, devices[index]) for index in visible),
    ]
    if not visible:
        lines.append("No selectable NVIDIA GPU IDs detected.")
    try:
        import psutil
        memory = psutil.virtual_memory()
        lines.append(f"Memory  {memory.available / 1024**3:.1f}/{memory.total / 1024**3:.1f} GiB available")
    except (ImportError, OSError):
        lines.append("Memory  unavailable")
    lines.extend([_disk_line("Workspace", workspace.root), _disk_line("Worker cache", workspace.cache)])
    for line in lines:
        _write(line, literal=True)
    return 0


def _runner_script(name: str) -> Path:
    script = PROJECT_ROOT / "esd_experiment" / name
    if not script.is_file():
        raise ValueError("This workflow needs the Model-Zoo source checkout. "
                         "From the checkout, run python -m zoo.tui with the same workspace.")
    return script


def _model_counts(path: Path) -> tuple[int, dict[str, int]]:
    counts: dict[str, int] = {}
    total = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if "model_id" not in (rows.fieldnames or []):
            raise ValueError(f"{path} needs a model_id column with one Hugging Face repository per entry.")
        for row in rows:
            if (row["model_id"] or "").strip() in {"", "nan"}:
                continue
            total += 1
            status = row.get("preparation_status") or "unreviewed"
            counts[status] = counts.get(status, 0) + 1
    return total, counts


def _entry_count(count: int) -> str:
    return f"{count:,} model {'entry' if count == 1 else 'entries'}"


def _prepare(workspace: Workspace) -> int:
    script = _runner_script("run_experiment.py")
    models = workspace.models
    if not models.is_file():
        models = _ask_path("Model list CSV", str(models),
                           detail="Put a CSV here or enter its path. Use a model_id column with "
                                  "one Hugging Face repository per row, such as org/model.")
    total, _ = _model_counts(models)
    if not total:
        raise ValueError(f"{models} has no model entries.")
    run = workspace.run
    if (run / "models.csv").exists():
        raise ValueError(f"{run} already has a pinned models.csv. Use Measure to resume it or "
                         "choose a fresh output folder with --run or set run in model-zoo.json.")
    command = [sys.executable, str(script), "--model_list", str(models),
               "--output_dir", str(run), "--prepare_only"]
    selected = total
    if workspace.prepare_limit is not None:
        command.extend(["--limit", str(workspace.prepare_limit)])
        selected = min(total, workspace.prepare_limit)
    return _run(command, workspace, label="Preparing model list", summary=[
        f"Input  {models} · {_entry_count(total)}",
        f"Reviewing {selected:,} of {total:,} entries in file order. Only Hub metadata is fetched.",
        f"Pinned revisions and ready/held reasons will be written to {run / 'models.csv'}.",
    ])


def _has_evidence(run: Path) -> bool:
    from evidence.compatibility import companion_manifest_path

    return any(not path.parent.name.startswith(".") and companion_manifest_path(path).is_file()
               for path in (run / "evidence").glob("*/*.fingerprint.safetensors"))


def _select_run(workspace: Workspace, *, purpose: str, matches, detail: str) -> Path:
    if matches(workspace.run):
        return workspace.run
    if workspace.run_is_explicit:
        raise ValueError(f"{workspace.run}: {detail}")
    runs_dir = workspace.root / "runs"
    candidates = sorted(run for run in runs_dir.iterdir() if matches(run)) if runs_dir.is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        _write("Available runs", "heading")
        for number, run in enumerate(candidates, 1):
            _write(f"{number}  {run}", literal=True)
        answer = _ask(purpose, detail=f"{detail} Enter a number or a full path.", completion="path")
        if answer.isascii() and answer.isdecimal() and len(answer) <= len(str(len(candidates))):
            index = int(answer) - 1
            if 0 <= index < len(candidates):
                return candidates[index]
        if not answer:
            raise ValueError("Choose a run number or enter its path.")
        run = _path(answer)
    else:
        run = _ask_path(purpose, str(workspace.run), detail=detail)
    if not matches(run):
        raise ValueError(f"{run}: {detail}")
    return run


def _launch(workspace: Workspace) -> int:
    if os.name == "nt":
        raise ValueError("GPU measurement needs the Linux process supervisor. "
                         "Use WSL or a Linux GPU host for this workflow.")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if any(item.strip().startswith(("GPU-", "MIG-")) for item in visible.split(",")):
        raise ValueError("The measurement runner needs an allocation with numeric GPU IDs. "
                         "This allocation uses GPU UUIDs or MIG IDs that cannot be passed safely to workers.")
    script = _runner_script("run_experiment.py")
    run = _select_run(workspace, purpose="Prepared run folder",
                      matches=lambda path: (path / "models.csv").is_file(),
                      detail="Choose a folder containing the pinned models.csv produced by Prepare.")
    model_list = run / "models.csv"
    total, counts = _model_counts(model_list)
    if not total:
        raise ValueError(f"{model_list} has no model entries.")
    _write("Reading local GPU availability…", flush=True)
    devices = _gpu_inventory()
    available = _gpu_choices(devices)
    gpu_ids = list(workspace.gpus)
    if available:
        for index in available:
            _write(_gpu_line(index, devices[index]) if index in devices
                   else f"GPU {index}  listed by CUDA_VISIBLE_DEVICES · live status unavailable")
        if not gpu_ids and len(available) == 1:
            gpu_ids = available
    if gpu_ids and set(gpu_ids) - set(available) and (devices or "CUDA_VISIBLE_DEVICES" in os.environ):
        raise ValueError("Configured gpus must be among the detected, visible NVIDIA GPU IDs: "
                         + (", ".join(map(str, available)) or "none"))
    if not gpu_ids:
        if not available:
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                raise ValueError("CUDA_VISIBLE_DEVICES exposes no identifiable full GPU. "
                                 "Run Scan to check the allocation on this host.")
            raise ValueError("No selectable NVIDIA GPU IDs detected. Run Scan to check this host, "
                             "or configure gpus in model-zoo.json if you know the allocated IDs.")
        while not gpu_ids:
            answer = _ask("GPU IDs allocated to this run",
                          detail="Use the IDs shown above, separated by spaces or commas. "
                                 "Example: 0, 2. The runner checks for idle devices before starting.")
            try:
                selected = [int(value) for value in answer.replace(",", " ").split()]
            except ValueError:
                selected = []
            if selected and len(set(selected)) == len(selected) and set(selected) <= set(available):
                gpu_ids = selected
            else:
                _write("Choose one or more distinct GPU IDs from the list.", "error")

    command = [sys.executable, str(script), "--model_list", str(model_list),
               "--output_dir", str(run), "--gpus", *map(str, gpu_ids),
               "--worker_cache_root", str(workspace.cache)]
    command.extend(f"--{key}={value}" for key, value in workspace.resources.items())
    options = {"max_concurrent_jobs": 1, "num_gpus_per_job": "auto", **workspace.resources}
    return _run(command, workspace, label="Measure pinned models", confirm=True, summary=[
        f"Run  {run}",
        f"Pinned list  {_entry_count(total)} · " + ", ".join(f"{count:,} {status}" for status, count in sorted(counts.items())),
        f"GPU pool  {', '.join(map(str, gpu_ids))} · " +
        ", ".join(f"{key.replace('_', ' ')}: {value}" for key, value in options.items()),
        *(_gpu_line(index, devices[index]) for index in gpu_ids if index in devices),
        _disk_line("Worker cache", workspace.cache),
        "The runner waits for idle GPUs, downloads weights as needed and publishes completed evidence.",
    ])


def _has_run_records(run: Path) -> bool:
    return (run / "models.csv").is_file() or any(
        any((run / directory).glob(pattern))
        for directory, pattern in (("stats", "*.csv"), ("metrics", "*.h5"), ("logs/terminal_status", "*.json"))
    )


def _analyze(workspace: Workspace) -> int:
    script = _runner_script("analyze_results.py")
    run = _select_run(workspace, purpose="Run to summarize",
                      matches=_has_run_records,
                      detail="Choose a run folder containing models.csv, measurements, or recorded worker outcomes.")
    return _run([sys.executable, str(script), "--results_dir", str(run)], workspace,
                label="Summarizing measurements", summary=[f"Output  {run / 'summary.csv'}"])


def _publish(workspace: Workspace) -> int:
    run = _select_run(workspace, purpose="Run to publish", matches=_has_evidence,
                      detail="Choose a completed run with manifests and fingerprints inside evidence/.")
    workspace.ensure_root()
    return _run([sys.executable, "-m", "zoo.cli", "publish", "--human", "--runs", str(run),
                 "--data_root", str(workspace.root)], workspace, label="Publishing completed evidence",
                summary=[f"Input  {run}", f"Model packages  {workspace.root / 'packages'}"])


def _build(workspace: Workspace) -> int:
    packages = workspace.root / "packages"
    if not any(packages.rglob("package.json")):
        raise ValueError(f"No model packages found in {packages}. Publish completed evidence first "
                         "or point --workspace at an existing Zoo data directory.")
    return _run([sys.executable, "-m", "zoo.cli", "build", "--data_root", str(workspace.root)],
                workspace, label="Building the searchable catalog",
                summary=[f"Reading packages from {packages}", f"Catalog  {workspace.root / 'builds' / 'current'}"])


def _catalog(workspace: Workspace) -> Path:
    path = workspace.root / "builds" / "current"
    if not (path / "models.parquet").is_file():
        raise ValueError(f"No ready catalog at {path}. Build the catalog or choose an existing workspace.")
    return path


def _deltas(workspace: Workspace) -> int:
    catalog = _catalog(workspace)
    return _run([sys.executable, "-m", "zoo.cli", "deltas", "--human", "--data_root", str(workspace.root),
                 "--workers", str(workspace.delta_workers), "--compute_device", workspace.delta_device],
                workspace, label="Measuring candidate model changes", confirm=True, summary=[
                    f"Catalog  {catalog}",
                    f"Compute device  {workspace.delta_device} · workers  {workspace.delta_workers}",
                ])


def _query(workspace: Workspace) -> int:
    catalog = _catalog(workspace)
    query_root = QUERY_ROOT if (QUERY_ROOT / "zoo_keeper").is_dir() else PROJECT_ROOT
    if query_root == PROJECT_ROOT and importlib.util.find_spec("zoo_keeper") is None:
        raise ValueError("Install the companion Zoo-Keeper package to search this catalog.")
    while True:
        choice = _ask_choice("Search by name or find similar fingerprints?", ("1", "2"), "1",
                             detail="1  Name or metadata     2  Similar weight fingerprints")
        if sys.stdin.isatty() and sys.stdout.isatty():
            _workflow_header("7")
        try:
            if choice == "1":
                term = _ask("Search text", detail="Enter a model name, organization or metadata keyword.")
                if not term:
                    return 0
                options = ["find", "--zoo", str(catalog), "--text=" + term, "--limit", str(workspace.find_limit)]
            else:
                model = _ask("Model ID", detail="Enter a repository like org/model or an exact org/model@revision.")
                if not model:
                    return 0
                options = ["similar", "--zoo", str(catalog), "-k", str(workspace.similar_limit), "--", model]
        except Back:
            _workflow_header("7")
            continue
        break
    command = [sys.executable, "-m", "zoo_keeper.cli", *options]
    return _run(command, workspace, label="Searching the catalog", cwd=query_root)


def _pause() -> None:
    if _page is None:
        _write(branded_footer(_layout().width, "Esc Back"), "muted", wrap=False)
    try:
        _read_input("\n" + _layout().indent + "Enter or Esc  Back to workflows")
    except (Back, EOFError, KeyboardInterrupt):
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


def _release_header() -> None:
    global _page
    if _page is not None:
        _, height = _page.size
        _page = None
        if height >= 3:
            sys.stdout.write(f"\x1b7\x1b[{height};1H\x1b[2K\x1b[r\x1b8")
            sys.stdout.flush()


def _clear_screen() -> None:
    _release_header()
    if not (sys.stdin.isatty() and sys.stdout.isatty()) or os.environ.get("TERM") == "dumb":
        return
    sys.stdout.flush()
    if os.name == "nt" and not os.environ.get("WT_SESSION"):
        try:
            subprocess.run(["cmd.exe", "/d", "/c", "cls"], check=False)
        except OSError:
            pass
    elif os.name == "nt" or os.environ.get("TERM"):
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.flush()


def _paint_star(char: str, role: str, *, animated: bool) -> str:
    family, level_text = role.split("_")
    level = int(level_text)
    if THEME.enabled:
        light = THEME.palette == "light"
        tint = 1 if family == "dust0" else 2
        if os.environ.get("WT_SESSION") or os.environ.get("COLORTERM", "").lower() in {"truecolor", "24bit"}:
            red, green, blue = star_rgb(light, tint, level)
            return f"\x1b[38;2;{red};{green};{blue}m{char}\x1b[0m"
        color = star_color(light, tint, level)
        return f"\x1b[38;5;{color}m{char}\x1b[0m"
    if animated:
        weight = 2 if level < 24 else 1 if level > 48 else 22
        return f"\x1b[{weight}m{char}\x1b[0m"
    return char


def _refresh_header() -> None:
    state = _page
    if state is None:
        return
    columns, height = state.size
    layout = PageLayout(columns)
    lines = brand_header(layout.width, height, CLOCK.tick(), max_rows=state.header_budget, star_seed=STAR_SEED)
    stars = {(row, column): _paint_star(line.text[column], role, animated=True)
             for row, line in enumerate(lines) for column, role in line.highlights}
    changes = []
    for row, column in stars.keys() | state.stars.keys():
        cell = stars.get((row, column), " ")
        if cell != state.stars.get((row, column)):
            changes.append(f"\x1b[{row + 1};{layout.gutter + column + 1}H{cell}")
    if changes:
        sys.stdout.write("\x1b7" + "".join(changes) + "\x1b8")
        sys.stdout.flush()
    state.stars.clear()
    state.stars.update(stars)


def _resize_page() -> None:
    previous = _page
    if previous is None:
        return
    _clear_screen()
    _draw_brand(reserve_rows=previous.reserve_rows, footer_hint=previous.footer_hint)
    if _page is None:
        return
    _page.body.extend(previous.body)
    lines = [line for entry in previous.body for line in _formatted_lines(*entry)]
    available = _page.size[1] - _page.header_rows - 2
    if available > 0:
        for line in lines[-available:]:
            sys.stdout.write(line + "\r\n")


def _draw_brand(*, reserve_rows: int = 0, footer_hint: str = "Esc Back") -> bool:
    if not (sys.stdin.isatty() and sys.stdout.isatty()) or os.environ.get("TERM") == "dumb":
        return False
    global _page
    from zoo.terminal_home import artwork_role

    columns, height = shutil.get_terminal_size((84, 24))
    layout = PageLayout(columns)
    vt = bool(os.environ.get("WT_SESSION") if os.name == "nt" else os.environ.get("TERM"))
    budget = max(1, height - max(reserve_rows, 4))
    lines = brand_header(layout.width, height, CLOCK.tick(), max_rows=budget, star_seed=STAR_SEED)
    if vt and height < 4:
        lines = []
    animated = CLOCK.started is not None and vt
    for line in lines:
        highlights = dict(line.highlights)

        def style(entry: tuple[int, str]) -> str:
            index, char = entry
            if char == " ":
                return "text"
            if index in highlights:
                return highlights[index]
            if line.artwork_start is not None and index >= line.artwork_start:
                return artwork_role(index - line.artwork_start, layout.width - line.artwork_start)
            return line.role

        spans = []
        for role, entries in groupby(enumerate(line.text.rstrip()), key=style):
            value = "".join(char for _, char in entries)
            spans.append(_paint_star(value, role, animated=animated) if role.startswith("dust") else THEME.paint(value, role))
        sys.stdout.write((layout.indent + "".join(spans) if line.text.strip() else "") + "\r\n")
    if vt:
        if height >= 3:
            top = len(lines) + 1
            footer = THEME.paint(branded_footer(layout.width, footer_hint), "muted")
            sys.stdout.write(f"\x1b[{top};{height - 1}r\x1b[{height};1H{layout.indent}{footer}\x1b[{top};1H")
        stars = {(row, col): _paint_star(line.text[col], role, animated=animated)
                 for row, line in enumerate(lines) for col, role in line.highlights}
        _page = TerminalPage((columns, height), reserve_rows, len(lines), budget, footer_hint, stars, deque(maxlen=80))
    sys.stdout.flush()
    return True


def _draw_home(root: Path | None) -> None:
    """Static, accessible fallback for terminals without curses support."""
    from zoo.terminal_home import WORKFLOWS

    status = _status(root)
    # Leave room for the status, menu, and its input prompt.
    if not _draw_brand(reserve_rows=len(status) + len(WORKFLOWS) + 4, footer_hint="1–8 Open · Esc Quit"):
        print()
        _write("MODEL ZOO", "brand")
    for line in status:
        _write(line, "muted", wrap=False)
    print()
    _write("Workflows", "accent")
    for item in WORKFLOWS:
        _write(f"{item.key}  {item.title}", wrap=False)
    if _page is None:
        _write(branded_footer(_layout().width, "Enter a number  ·  Esc or Q Quit"), "muted", wrap=False)


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


def _workflow_header(choice: str) -> None:
    from zoo.terminal_home import WORKFLOWS

    item = next(item for item in WORKFLOWS if item.key == choice)
    _clear_screen()
    if _draw_brand():
        _write(item.group, "workflow")
    else:
        print()
        _write(f"Model Zoo / {item.group}", "workflow")
    _write(item.title, "heading")
    _write(item.description, "text")
    if _page is None:
        _write(branded_footer(_layout().width, "Esc Back"), "muted", wrap=False)
    print()


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main(argv)
    finally:
        _release_header()
        CLOCK.set_running(False)


def _main(argv: Sequence[str] | None) -> int:
    global VERBOSE, CLOCK
    actions = {
        "prepare": _prepare, "measure": _launch, "summarize": _analyze,
        "publish": _publish, "build": _build, "deltas": _deltas,
        "search": _query, "scan": _infrastructure,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", "--data-root", type=Path, metavar="DIR",
                        help="Workspace folder (default: ZOO_DATA or ./zoo-data)")
    parser.add_argument("--run", type=Path, metavar="DIR",
                        help="Use a specific run folder, relative to the workspace or absolute")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show the underlying command")
    parser.add_argument("--theme", choices=("auto", "dark", "light"), default="auto",
                        help="Terminal background palette (default: auto)")
    animation = parser.add_mutually_exclusive_group()
    animation.add_argument("--motion", action="store_true", help="Animate stars on menus and workflow prompts (default)")
    animation.add_argument("--no-motion", action="store_false", dest="motion",
                           help="Keep the stars still")
    parser.set_defaults(motion=True)
    parser.add_argument("--plain", action="store_true", help="Use the numbered menu")
    parser.add_argument("workflow", nargs="?", choices=("init", *actions),
                        help="Open a workflow directly, or use init to create an editable config")
    args = parser.parse_args(argv)
    CLOCK = MotionClock()
    CLOCK.set_running(args.motion)
    VERBOSE = args.verbose
    THEME.palette = THEME.automatic_palette() if args.theme == "auto" else args.theme
    THEME.enabled = Theme().enabled and (os.name != "nt" or bool(os.environ.get("WT_SESSION")))
    supplied = args.workspace or Path(os.environ.get("ZOO_DATA") or "zoo-data")
    try:
        root = supplied.expanduser().resolve()
        if root.exists() and not root.is_dir():
            raise ValueError(f"Workspace must be a directory: {root}")
        workspace = Workspace.load(root)
        if args.run is not None:
            workspace = replace(workspace, run=(root / args.run.expanduser()).resolve(), run_is_explicit=True)
        if args.workflow == "init":
            config = workspace.initialize()
            _clear_screen()
            _draw_brand()
            _write(f"Workspace  {root}", literal=True)
            _write(f"Settings  {config}", literal=True)
            _write(f"Model list  {workspace.models}", literal=True)
            _write("For a new collection, add a CSV with a model_id column at the model list path.")
            return 0
    except (OSError, ValueError, RuntimeError) as error:
        _write(str(error), "error")
        return 2

    def run_workflow(workflow: str) -> int | None:
        try:
            return actions[workflow](workspace)
        except Back:
            return None
        except KeyboardInterrupt:
            _write("\nWorkflow interrupted.", "muted")
            return 130
        except EOFError:
            _write("\nInput ended before the workflow could finish.", "error")
            return 2
        except (OSError, ValueError, csv.Error) as error:
            _write(str(error), "error")
            return 2

    by_number = {str(index): workflow for index, workflow in enumerate(actions, 1)}
    if args.workflow:
        if sys.stdin.isatty() and sys.stdout.isatty():
            choice = next(key for key, workflow in by_number.items() if workflow == args.workflow)
            _workflow_header(choice)
        return run_workflow(args.workflow) or 0

    motion = args.motion
    full_screen = (not args.plain and (os.name != "nt" or bool(os.environ.get("WT_SESSION")))
                   and sys.stdin.isatty()
                   and sys.stdout.isatty() and _supports_full_screen())
    if full_screen:
        try:
            import curses
            from zoo.terminal_home import navigate
        except ImportError:
            full_screen = False

    refresh = True
    while True:
        try:
            if refresh:
                _clear_screen()
            if full_screen:
                try:
                    choice, THEME.palette, motion = navigate(_status(root), palette=THEME.palette,
                                                              motion=motion, color=THEME.enabled, star_seed=STAR_SEED, clock=CLOCK)
                except curses.error:
                    full_screen = False
                    continue
            else:
                if refresh:
                    _draw_home(root)
                choice = _read_menu_choice(root)
            choice = choice.strip().casefold()
        except (Back, EOFError, KeyboardInterrupt):
            print("\nSee you next time.")
            return 0
        if choice in {"q", "quit", "exit", "\x1b"}:
            print("See you next time.")
            return 0
        workflow = by_number.get(choice)
        if workflow is None:
            _write("Choose a number from 1 to 8, or Esc to quit.", "error")
            refresh = False
            continue
        _workflow_header(choice)
        result = run_workflow(workflow)
        if result is not None and sys.stdin.isatty() and sys.stdout.isatty():
            _pause()
        refresh = True


if __name__ == "__main__":
    raise SystemExit(main())
