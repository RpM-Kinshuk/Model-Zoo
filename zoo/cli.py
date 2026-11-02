"""Build a queryable zoo from existing manifests, fingerprints and scores."""
import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd

from .build import build_zoo
from .schema import model_key
from .scores import load_leaderboard_v2, load_scores_csv


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Publish a complete artifact-only zoo snapshot")
    build.add_argument("--manifests", type=Path, required=True)
    build.add_argument("--fingerprints", type=Path, nargs="+", required=True)
    build.add_argument("--data_root", type=Path, help="Artifact root; defaults to ZOO_DATA")
    build.add_argument("--out", type=Path, help="Defaults to <data_root>/builds/current")
    for name in ("frame", "leaderboard_v2", "scores_csv", "scores_mapping", "lineage_edges", "null"):
        build.add_argument("--" + name, type=Path)
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
        if bool(args.scores_csv) != bool(args.scores_mapping):
            raise ValueError("--scores_csv and --scores_mapping must be supplied together")
        root = args.data_root or (Path(os.environ["ZOO_DATA"]) if os.environ.get("ZOO_DATA") else None)
        if root is None:
            raise ValueError("Set --data_root or ZOO_DATA")
        if not root.is_dir():
            raise ValueError("data_root must be an existing directory")
        if not args.manifests.is_dir():
            raise ValueError("--manifests must be an existing directory")
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
            models = _score_models(args.manifests)
            if args.leaderboard_v2:
                scores.append(load_leaderboard_v2(args.leaderboard_v2, models))
            if args.scores_csv:
                scores.append(load_scores_csv(args.scores_csv, mapping, models))
        published = build_zoo(args.manifests, args.fingerprints, out, data_root=root,
                              frame=args.frame, score_tables=scores,
                              lineage_edges=args.lineage_edges, null=args.null)
    except (OSError, ValueError, KeyError) as error:
        print(f"zoo build: {error}", file=sys.stderr)
        return 2
    print(f"Built zoo: {published}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
