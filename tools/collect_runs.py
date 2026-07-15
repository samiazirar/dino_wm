#!/usr/bin/env python3
"""Pool fixed-manifest squared errors and make the locked P2a decision."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


RESULT_SCHEMA = "dino-wm-open-loop-episode-errors-v1"
DECISION_SCHEMA = "dino-wm-p2a-producer-decision-v1"
PRODUCERS = ("da3_giant_video", "mapanything_recovered_framewise")
P2A_HORIZONS = (5, 10)
TIE_TOLERANCE = 1e-6
BOOTSTRAP_SEED = 20260714
BOOTSTRAP_REPLICATES = 10000
P4_ARMS = ("dino_pinned", "dinocular", "dinocular_zerodepth")
P4_SEEDS = (1, 2, 3)
P4_HORIZONS = {
    "pusht": (1, 5, 10, 25),
    "wall": (1, 5, 10),
    "rope": (1, 2, 3),
    "granular": (1, 2, 3),
}


class CollectionError(RuntimeError):
    """Coverage, finite-value, denominator, pairing, or decision gate failed."""


def _read(paths: Iterable[Path]) -> list[Mapping[str, Any]]:
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise CollectionError(f"blank JSONL line at {path}:{line_number}")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CollectionError(
                        f"invalid JSONL at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(row, Mapping) or row.get("schema") != RESULT_SCHEMA:
                    raise CollectionError(f"invalid evaluator record at {path}:{line_number}")
                rows.append(row)
    if not rows:
        raise CollectionError("no evaluator records were provided")
    return rows


def _values(row: Mapping[str, Any], horizon: int) -> tuple[float, float, int]:
    record = row.get("horizons", {}).get(str(horizon))
    if not isinstance(record, Mapping):
        raise CollectionError(
            f"missing horizon {horizon} for {row.get('run_id')}/episode {row.get('episode')}"
        )
    model = record.get("model_squared_error")
    persistence = record.get("persistence_squared_error")
    elements = record.get("element_count")
    if (
        not isinstance(model, (int, float))
        or not isinstance(persistence, (int, float))
        or not isinstance(elements, int)
        or not math.isfinite(float(model))
        or not math.isfinite(float(persistence))
        or model < 0
        or persistence < 0
        or elements <= 0
    ):
        raise CollectionError("nonfinite, negative, or empty squared-error record")
    return float(model), float(persistence), int(elements)


def _pool(rows: Sequence[Mapping[str, Any]], horizon: int) -> Mapping[str, Any]:
    model = persistence = 0.0
    elements = 0
    for row in rows:
        model_value, persistence_value, count = _values(row, horizon)
        model += model_value
        persistence += persistence_value
        elements += count
    denominator_mse = persistence / elements
    if denominator_mse <= 1e-12:
        raise CollectionError(
            f"pooled persistence MSE is degenerate at horizon {horizon}: {denominator_mse}"
        )
    nre = model / persistence
    if not math.isfinite(nre):
        raise CollectionError(f"pooled NRE is nonfinite at horizon {horizon}")
    return {
        "model_squared_error": model,
        "persistence_squared_error": persistence,
        "element_count": elements,
        "persistence_mse": denominator_mse,
        "nre": nre,
    }


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise CollectionError(f"immutable output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def producer_pilot(args: argparse.Namespace) -> None:
    if (
        args.bootstrap != BOOTSTRAP_REPLICATES
        or args.seed != BOOTSTRAP_SEED
        or args.tie_tolerance != TIE_TOLERANCE
    ):
        raise CollectionError("P2a bootstrap or tie rule differs from the locked contract")
    rows = _read(args.inputs)
    if any(
        row.get("environment") != "pusht"
        or int(row.get("seed", -1)) != 1
        or row.get("arm") != "dinocular"
        for row in rows
    ):
        raise CollectionError("P2a records must be PushT seed-1 depth-on DINOcular")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        producer = row.get("producer")
        if producer not in PRODUCERS:
            raise CollectionError(f"unregistered P2a producer {producer!r}")
        grouped[str(producer)].append(row)
    if tuple(grouped) != PRODUCERS and set(grouped) != set(PRODUCERS):
        raise CollectionError("P2a must contain exactly both locked producers")
    paired = {}
    manifest_hashes = set()
    for producer in PRODUCERS:
        rows_by_episode = {}
        keys = []
        for row in grouped[producer]:
            episode = int(row["episode"])
            if episode in rows_by_episode:
                raise CollectionError(f"duplicate P2a episode {producer}/{episode}")
            rows_by_episode[episode] = row
            keys.extend(row["manifest_keys"])
            manifest_hashes.add(row["manifest_sha256"])
        if len(keys) != len(set(keys)):
            raise CollectionError(f"duplicate fixed-manifest key for {producer}")
        paired[producer] = rows_by_episode
    episode_sets = [set(paired[producer]) for producer in PRODUCERS]
    key_sets = [
        {key for row in paired[producer].values() for key in row["manifest_keys"]}
        for producer in PRODUCERS
    ]
    if episode_sets[0] != episode_sets[1] or key_sets[0] != key_sets[1]:
        raise CollectionError("P2a producer coverage is not exactly paired")
    if len(manifest_hashes) != 1:
        raise CollectionError("P2a producers used different fixed manifest hashes")
    episodes = sorted(episode_sets[0])
    pooled = {}
    scores = {}
    for producer in PRODUCERS:
        pooled[producer] = {
            str(horizon): _pool(list(paired[producer].values()), horizon)
            for horizon in P2A_HORIZONS
        }
        scores[producer] = 0.5 * sum(
            pooled[producer][str(horizon)]["nre"] for horizon in P2A_HORIZONS
        )
    delta = scores[PRODUCERS[0]] - scores[PRODUCERS[1]]
    if abs(delta) <= args.tie_tolerance:
        raise CollectionError(
            f"P2a point estimates tie within {args.tie_tolerance}: delta={delta}"
        )
    winner = PRODUCERS[0] if scores[PRODUCERS[0]] < scores[PRODUCERS[1]] else PRODUCERS[1]

    rng = np.random.default_rng(args.seed)
    bootstrap = np.empty(args.bootstrap, dtype=np.float64)
    for replicate in range(args.bootstrap):
        sampled = rng.choice(episodes, size=len(episodes), replace=True)
        sample_scores = {}
        for producer in PRODUCERS:
            horizon_scores = []
            for horizon in P2A_HORIZONS:
                model = persistence = 0.0
                elements = 0
                for episode in sampled:
                    m, p, n = _values(paired[producer][int(episode)], horizon)
                    model += m
                    persistence += p
                    elements += n
                if persistence / elements <= 1e-12:
                    raise CollectionError("bootstrap replicate has degenerate denominator")
                horizon_scores.append(model / persistence)
            sample_scores[producer] = 0.5 * sum(horizon_scores)
        bootstrap[replicate] = sample_scores[PRODUCERS[0]] - sample_scores[PRODUCERS[1]]
    interval = np.percentile(bootstrap, [2.5, 97.5]).tolist()
    decision = {
        "schema": DECISION_SCHEMA,
        "status": "PASS",
        "environment": "pusht",
        "seed": 1,
        "arm": "dinocular",
        "manifest_sha256": next(iter(manifest_hashes)),
        "episode_count": len(episodes),
        "manifest_key_count": len(key_sets[0]),
        "decision_horizons": list(P2A_HORIZONS),
        "pooled": pooled,
        "scores": scores,
        "delta_da3_minus_mapanything": delta,
        "paired_trajectory_bootstrap": {
            "seed": args.seed,
            "replicates": args.bootstrap,
            "unit": "held_out_episode",
            "percentile_95": interval,
        },
        "tie_tolerance": args.tie_tolerance,
        "winner": winner,
        "selection_rule": "strictly_lower_locked_point_estimate",
    }
    _write_immutable(args.out, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


def _validate_open_loop_coverage(
    groups: Mapping[tuple[str, str, int], Sequence[Mapping[str, Any]]]
) -> Mapping[str, Mapping[str, Any]]:
    environments = sorted({key[0] for key in groups})
    if not environments or any(environment not in P4_HORIZONS for environment in environments):
        raise CollectionError("P4 contains no environment or an unknown environment")
    contracts = {}
    for environment in environments:
        expected_groups = {
            (environment, arm, seed) for arm in P4_ARMS for seed in P4_SEEDS
        }
        actual_groups = {key for key in groups if key[0] == environment}
        if actual_groups != expected_groups:
            raise CollectionError(
                f"P4 must contain exactly all 9 arm/seed groups for {environment}"
            )
        group_contracts = {}
        for key in sorted(expected_groups):
            rows = groups[key]
            episodes = []
            coverage = []
            hashes = set()
            for row in rows:
                row_horizons = {int(value) for value in row.get("horizons", {})}
                if row_horizons != set(P4_HORIZONS[environment]):
                    raise CollectionError(f"P4 horizon set differs for {key}")
                keys = row.get("manifest_keys")
                if not isinstance(keys, list) or not keys or len(keys) != len(set(keys)):
                    raise CollectionError(f"P4 manifest-key record is empty or duplicated for {key}")
                episodes.append(int(row["episode"]))
                coverage.extend(str(value) for value in keys)
                hashes.add(str(row.get("manifest_sha256")))
            if (
                len(episodes) != len(set(episodes))
                or len(coverage) != len(set(coverage))
                or len(hashes) != 1
            ):
                raise CollectionError(f"duplicate coverage or manifest hashes within {key}")
            group_contracts[key] = {
                "episodes": frozenset(episodes),
                "keys": frozenset(coverage),
                "manifest_sha256": next(iter(hashes)),
            }
        reference = next(iter(group_contracts.values()))
        for key, contract in group_contracts.items():
            if contract != reference:
                raise CollectionError(
                    f"P4 manifest hash, exact keys, or episode coverage differs for {key}"
                )
        contracts[environment] = {
            **reference,
            "horizons": P4_HORIZONS[environment],
        }
    return contracts


def open_loop(args: argparse.Namespace) -> None:
    if args.bootstrap != BOOTSTRAP_REPLICATES or args.seed != BOOTSTRAP_SEED or not args.paired:
        raise CollectionError("P4 requires the locked paired 10000-replicate bootstrap")
    rows = _read(args.inputs)
    groups: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["environment"]), str(row["arm"]), int(row["seed"]))
        groups[key].append(row)
    coverage_contracts = _validate_open_loop_coverage(groups)
    summaries = {}
    for key, values in sorted(groups.items()):
        environment, arm, seed = key
        episodes = [int(row["episode"]) for row in values]
        coverage = [manifest_key for row in values for manifest_key in row["manifest_keys"]]
        if len(episodes) != len(set(episodes)) or len(coverage) != len(set(coverage)):
            raise CollectionError(f"duplicate episode or manifest key for {key}")
        horizons = list(coverage_contracts[environment]["horizons"])
        summaries[f"{environment}/{arm}/s{seed}"] = {
            "episode_count": len(episodes),
            "manifest_key_count": len(coverage),
            "manifest_sha256": values[0]["manifest_sha256"],
            "horizons": {
                str(horizon): _pool(values, horizon) for horizon in horizons
            },
        }

    environments = sorted({key[0] for key in groups})
    contrasts = {}
    rng = np.random.default_rng(args.seed)
    for environment in environments:
        env_keys = [key for key in groups if key[0] == environment]
        arms = list(P4_ARMS)
        seeds = list(P4_SEEDS)
        episode_sets = {
            (arm, seed): {int(row["episode"]) for row in groups[(environment, arm, seed)]}
            for arm in arms
            for seed in seeds
        }
        common_episodes = set.intersection(*episode_sets.values())
        if any(value != common_episodes for value in episode_sets.values()):
            raise CollectionError(f"P4 episode coverage is not paired for {environment}")
        by_arm_seed_episode = {
            (arm, seed, int(row["episode"])): row
            for arm in arms
            for seed in seeds
            for row in groups[(environment, arm, seed)]
        }
        horizons = list(coverage_contracts[environment]["horizons"])
        for left_index, left in enumerate(arms):
            for right in arms[left_index + 1 :]:
                for horizon in horizons:
                    values = np.empty(args.bootstrap, dtype=np.float64)
                    for replicate in range(args.bootstrap):
                        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
                        sampled_episodes = rng.choice(
                            sorted(common_episodes),
                            size=len(common_episodes),
                            replace=True,
                        )
                        scores = {}
                        for arm in (left, right):
                            model = persistence = 0.0
                            elements = 0
                            for seed in sampled_seeds:
                                for episode in sampled_episodes:
                                    m, p, n = _values(
                                        by_arm_seed_episode[
                                            (arm, int(seed), int(episode))
                                        ],
                                        horizon,
                                    )
                                    model += m
                                    persistence += p
                                    elements += n
                            if persistence / elements <= 1e-12:
                                raise CollectionError(
                                    "crossed bootstrap has degenerate denominator"
                                )
                            scores[arm] = model / persistence
                        values[replicate] = scores[left] - scores[right]
                    contrasts[f"{environment}/{left}_minus_{right}/h{horizon}"] = {
                        "bootstrap_seed": args.seed,
                        "replicates": args.bootstrap,
                        "resampling": "crossed_training_seed_and_episode",
                        "percentile_95": np.percentile(values, [2.5, 97.5]).tolist(),
                    }
    output = {
        "schema": "dino-wm-open-loop-collection-v1",
        "state": "PASS",
        "summaries": summaries,
        "paired_contrasts": contrasts,
    }
    _write_immutable(args.out, output)
    print(json.dumps(output, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pilot = subparsers.add_parser("producer-pilot")
    pilot.add_argument("--inputs", type=Path, nargs="+", required=True)
    pilot.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPLICATES)
    pilot.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    pilot.add_argument("--tie-tolerance", type=float, default=TIE_TOLERANCE)
    pilot.add_argument("--out", type=Path, required=True)
    pilot.set_defaults(function=producer_pilot)
    pooled = subparsers.add_parser("open-loop")
    pooled.add_argument("--inputs", type=Path, nargs="+", required=True)
    pooled.add_argument("--bootstrap", type=int, required=True)
    pooled.add_argument("--paired", action="store_true")
    pooled.add_argument("--seed", type=int, required=True)
    pooled.add_argument("--out", type=Path, required=True)
    pooled.set_defaults(function=open_loop)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.function(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CollectionError as exc:
        print(f"COLLECTION CONTRACT FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(2)
