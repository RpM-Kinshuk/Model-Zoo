"""Build a queryable zoo from existing manifests, fingerprints and scores."""
import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd

from .build import build_zoo
from .schema import model_key
from .publish import publish_packages, package_inputs, current_conventions_id
from .scores import load_leaderboard_v2, load_scores_csv


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Publish a complete artifact-only zoo snapshot")
    build.add_argument("--manifests", type=Path)
    build.add_argument("--fingerprints", type=Path, nargs="+")
    build.add_argument("--packages", type=Path)
    build.add_argument("--conventions", dest="conventions_id")
    build.add_argument("--data_root", type=Path, help="Artifact root; defaults to ZOO_DATA")
    build.add_argument("--out", type=Path, help="Defaults to <data_root>/builds/current")
    for name in ("frame", "leaderboard_v2", "scores_csv", "scores_mapping", "lineage_edges", "null"):
        build.add_argument("--" + name, type=Path)
    publish = commands.add_parser("publish", help="Backfill immutable model packages from completed runs")
    publish.add_argument("--runs", type=Path, nargs="+", required=True)
    publish.add_argument("--packages_root", type=Path)
    publish.add_argument("--data_root", type=Path)
    return parser


def _score_models(directory):
    # Score adapters need pinned identities, not tensors or a second table build.
    rows = []
    for path in sorted(directory.glob("*.manifest.json")):
        manifest = json.loads(path.read_text())
        rows.append(dict(model_key=model_key(manifest["repo_id"], manifest["revision"]),
                         repo_id=manifest["repo_id"], revision=manifest["revision"]))
    return pd.DataFrame(rows, columns=["model_key", "repo_id", "revision"])


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.command == "publish":
            root = args.data_root or (Path(os.environ["ZOO_DATA"]) if os.environ.get("ZOO_DATA") else None)
            if root is None:
                raise ValueError("Set --data_root or ZOO_DATA")
            published = publish_packages(args.runs, args.packages_root or root / "packages", root)
            print(json.dumps({"packages": [str(path) for path in published], "count": len(published)}, sort_keys=True))
            return 0
        if bool(args.scores_csv) != bool(args.scores_mapping):
            raise ValueError("--scores_csv and --scores_mapping must be supplied together")
        root = args.data_root or (Path(os.environ["ZOO_DATA"]) if os.environ.get("ZOO_DATA") else None)
        if root is None:
            raise ValueError("Set --data_root or ZOO_DATA")
        if not root.is_dir():
            raise ValueError("data_root must be an existing directory")
        package_mode = args.packages is not None or (args.manifests is None and args.fingerprints is None)
        if package_mode:
            if args.manifests is not None or args.fingerprints is not None:
                raise ValueError("Choose --packages or --manifests/--fingerprints")
            args.packages = args.packages or root / "packages"
            args.conventions_id = args.conventions_id or current_conventions_id()
        else:
            if args.manifests is None or not args.manifests.is_dir() or not args.fingerprints:
                raise ValueError("Supply --manifests and --fingerprints together")
            if any(not path.exists() for path in args.fingerprints):
                raise ValueError("Every --fingerprints input must exist")
        out = args.out or root / "builds" / "current"
        mapping = None
        if args.scores_mapping:
            mapping = json.loads(args.scores_mapping.read_text())
            if not isinstance(mapping, dict):
                raise ValueError("--scores_mapping must contain a JSON object")
        scores = []
        if args.leaderboard_v2 or args.scores_csv:
            if package_mode:
                selected, _ = package_inputs(args.packages, args.conventions_id)
                models = pd.DataFrame([dict(model_key=meta["model_key"], repo_id=meta["repo_id"], revision=meta["revision"]) for _, meta in selected],
                                      columns=["model_key", "repo_id", "revision"])
            else:
                models = _score_models(args.manifests)
            if args.leaderboard_v2:
                scores.append(load_leaderboard_v2(args.leaderboard_v2, models))
            if args.scores_csv:
                scores.append(load_scores_csv(args.scores_csv, mapping, models))
        options = dict(frame=args.frame, score_tables=scores, lineage_edges=args.lineage_edges, null=args.null)
        if package_mode:
            options.update(packages=args.packages, conventions_id=args.conventions_id)
        published = build_zoo(args.manifests, args.fingerprints, out, data_root=root, **options)
    except (OSError, ValueError, KeyError) as error:
        print(f"zoo build: {error}", file=sys.stderr)
        return 2
    print(f"Built zoo: {published}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
