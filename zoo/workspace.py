"""Paths and optional overrides for the guided Model Zoo launcher."""

from dataclasses import dataclass
import json
import os
from pathlib import Path


RESOURCE_OPTIONS = (
    "num_gpus_per_job", "max_concurrent_jobs", "max_weight_gib",
    "max_worker_ram_gib", "max_worker_cache_gib", "cpu_threads_per_gpu",
    "prefetch_models", "prefetch_max_gib", "gpu_memory_threshold", "max_check",
    "stale_process_action", "heartbeat_timeout_seconds", "stage_timeout_seconds",
    "termination_grace_seconds",
)


@dataclass(frozen=True)
class Workspace:
    root: Path
    models: Path
    run: Path
    run_is_explicit: bool
    cache: Path
    prepare_limit: int | None
    gpus: tuple[int, ...]
    resources: dict[str, str | int | float]
    find_limit: int
    similar_limit: int
    delta_workers: int | str
    delta_device: str

    @staticmethod
    def config_path(root: Path) -> Path:
        return root / "model-zoo.json"

    @classmethod
    def load(cls, root: Path) -> "Workspace":
        config = cls.config_path(root)
        try:
            settings = json.loads(config.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            settings = {}
        except ValueError as error:
            raise ValueError(f"Invalid JSON in {config}: {error}") from error
        allowed = {"models", "run", "cache", "prepare_limit", "gpus", "resources",
                   "find_limit", "similar_limit", "delta_workers", "delta_device"}
        if not isinstance(settings, dict):
            raise ValueError(f"{config} must contain a JSON object.")
        unknown = settings.keys() - allowed
        if unknown:
            raise ValueError(f"Unknown settings in {config}: {', '.join(sorted(unknown))}")

        def path(key, default):
            value = settings.get(key, default)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{config}: {key} must be a nonempty path.")
            return (root / Path(value).expanduser()).resolve()

        def limit(key, default):
            value = settings.get(key, default)
            if value is None and default is None:
                return value
            if type(value) is not int or value < 1:
                raise ValueError(f"{config}: {key} must be a positive whole number.")
            return value

        gpus = settings.get("gpus", [])
        if not isinstance(gpus, list) or any(type(gpu) is not int or gpu < 0 for gpu in gpus):
            raise ValueError(f"{config}: gpus must be a list of nonnegative GPU IDs.")
        if len(set(gpus)) != len(gpus):
            raise ValueError(f"{config}: gpus must contain unique GPU IDs.")
        resources = settings.get("resources", {})
        if not isinstance(resources, dict) or resources.keys() - set(RESOURCE_OPTIONS):
            raise ValueError(f"{config}: resources accepts {', '.join(RESOURCE_OPTIONS)}.")
        if any(type(value) not in (str, int, float) for value in resources.values()):
            raise ValueError(f"{config}: resource values must be strings or numbers.")
        delta_workers = settings.get("delta_workers", "auto")
        if delta_workers != "auto" and (type(delta_workers) is not int or delta_workers < 1):
            raise ValueError(f"{config}: delta_workers must be auto or a positive whole number.")
        delta_device = settings.get("delta_device", "auto")
        if delta_device not in ("auto", "cpu", "cuda"):
            raise ValueError(f"{config}: delta_device must be auto, cpu or cuda.")
        cache = os.environ.get("MODEL_ZOO_WORKER_CACHE_ROOT")
        return cls(
            root=root,
            models=path("models", "models.csv"),
            run=path("run", "runs/default"),
            run_is_explicit=settings.get("run", "runs/default") != "runs/default",
            cache=path("cache", str(Path(cache).expanduser().resolve()) if cache else "cache/workers"),
            prepare_limit=limit("prepare_limit", None),
            gpus=tuple(gpus),
            resources=resources,
            find_limit=limit("find_limit", 10),
            similar_limit=limit("similar_limit", 5),
            delta_workers=delta_workers,
            delta_device=delta_device,
        )

    def ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> Path:
        self.ensure_root()
        (self.root / "runs").mkdir(exist_ok=True)
        config = self.config_path(self.root)
        try:
            with config.open("x", encoding="utf-8") as stream:
                json.dump({"models": "models.csv", "run": "runs/default", "prepare_limit": None,
                           "gpus": [], "resources": {}, "find_limit": 10, "similar_limit": 5,
                           "delta_workers": "auto", "delta_device": "auto"},
                          stream, indent=2)
                stream.write("\n")
        except FileExistsError:
            pass
        return config
