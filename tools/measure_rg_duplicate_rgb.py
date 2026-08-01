#!/usr/bin/env python3
"""Classify every receipt-counted corrected-depth duplicate by aligned RGB."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import torch


ENVIRONMENTS = ("rope", "granular")
RGB_MOVING_THRESHOLD_UINT8 = 5.0
CHECKPOINT_SHA256 = (
    "decc7c73283bf46f66dbbedec6fb065ad0a943fb2ddb1c49317fafb0319f5dcc"
)
PINNED_RGB_SHA256 = {
    "rope": "28cf9c38be5adb9cb5c67a12d41407807f8b523c630fd93331ece07c21a49439",
    "granular": "3bbcd4f293866d797ca8844fbd76f33ce401ebbabadd5334f7c5774063ca4d4c",
}


def canonical_sha256(value: dict[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("receipt_sha256", None)
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object at {path}")
    return value


def check_final_receipt(
    path: Path,
    environment: str,
    replacement_manifest: Path,
    defective_manifest: Path,
    expected_repeated: int,
) -> dict[str, Any]:
    receipt = load_json(path)
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or claimed != canonical_sha256(receipt):
        raise RuntimeError(f"{environment} final receipt self-hash is invalid")
    if receipt.get("schema") != "dinocular.depth-admission-validation.v1":
        raise RuntimeError(f"{environment} final receipt schema differs")
    if receipt.get("environment") != environment:
        raise RuntimeError(f"{environment} final receipt environment differs")

    artifacts = receipt.get("artifacts", {})
    actual_replacement_manifest_sha256 = sha256_file(replacement_manifest)
    actual_defective_manifest_sha256 = sha256_file(defective_manifest)
    if artifacts.get("cache_manifest_sha256") != actual_replacement_manifest_sha256:
        raise RuntimeError(f"{environment} final receipt replacement manifest binding differs")
    if artifacts.get("defective_cache_manifest_sha256") != actual_defective_manifest_sha256:
        raise RuntimeError(f"{environment} final receipt defective manifest binding differs")
    manifest = load_json(replacement_manifest)
    producer_sha256 = hashlib.sha256(
        json.dumps(
            manifest["producer"], sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    if artifacts.get("producer_sha256") != producer_sha256:
        raise RuntimeError(f"{environment} final receipt producer binding differs")
    if artifacts.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise RuntimeError(f"{environment} final receipt checkpoint binding differs")
    repeated = receipt.get("checks", {}).get("repeated_frame_hash_rate", {})
    if repeated.get("payloads") != 20000 or repeated.get("repeated_payloads") != expected_repeated:
        raise RuntimeError(f"{environment} final receipt duplicate count differs")
    if receipt.get("checks", {}).get("rgb_depth_frame_alignment", {}).get("state") != "PASS":
        raise RuntimeError(f"{environment} final receipt RGB/depth alignment is not PASS")
    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "receipt_sha256": claimed,
        "state": receipt.get("state"),
        "artifacts": artifacts,
        "repeated_payloads": repeated,
    }


def manifest_key_map(manifest: dict[str, Any], source_root: Path) -> dict[str, dict[str, Any]]:
    if manifest.get("schema") != "dinocular-depth-cache-v1":
        raise RuntimeError("corrected cache manifest schema differs")
    if manifest.get("closed_before_hash") is not True:
        raise RuntimeError("corrected cache manifest was not closed before hashing")
    if manifest.get("frame_count") != 20000 or manifest.get("trajectory_count") != 1000:
        raise RuntimeError("corrected cache manifest coverage differs")
    wire_format = manifest.get("wire_format", {})
    if wire_format.get("dtype") != "<f2" or wire_format.get("shape") != [224, 224]:
        raise RuntimeError("corrected cache wire format differs")
    if wire_format.get("normalization") != "none_raw_depth_z_float16":
        raise RuntimeError("corrected cache normalization differs")
    records: dict[str, dict[str, Any]] = {}
    for trajectory in manifest.get("trajectories", []):
        source_path = str(trajectory["source_path"])
        source_sha256 = str(trajectory["source_video_sha256"])
        for key in trajectory["ordered_output_keys"]:
            split, episode, frame = str(key).split("/")
            if key in records:
                raise RuntimeError(f"duplicate corrected manifest key {key}")
            records[key] = {
                "source_path": str((source_root / source_path).resolve()),
                "source_relative_path": source_path,
                "source_sha256": source_sha256,
                "split": split,
                "episode": int(episode),
                "frame": int(frame),
            }
    if len(records) != 20000:
        raise RuntimeError(f"corrected manifest key coverage is {len(records)}, not 20000")
    return records


def duplicate_groups(cache_dir: Path) -> tuple[dict[str, list[str]], int, int]:
    database = lmdb.open(
        str(cache_dir), readonly=True, lock=False, readahead=False, meminit=False
    )
    groups: dict[str, list[str]] = defaultdict(list)
    payloads = 0
    try:
        with database.begin(write=False) as transaction:
            for key, payload in transaction.cursor():
                payloads += 1
                digest = hashlib.sha256(payload).hexdigest()
                groups[digest].append(key.decode("ascii"))
    finally:
        database.close()
    repeated = sum(len(keys) - 1 for keys in groups.values() if len(keys) > 1)
    return groups, payloads, repeated


class RGBSourceCache:
    def __init__(self) -> None:
        self._arrays: dict[str, np.ndarray] = {}
        self._hashes: dict[str, str] = {}

    def load(self, reference: dict[str, Any]) -> np.ndarray:
        path = reference["source_path"]
        if path not in self._arrays:
            source = Path(path)
            actual_sha256 = sha256_file(source)
            expected_sha256 = reference["source_sha256"]
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"RGB source hash mismatch for {source}: "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )
            stored = torch.load(source, map_location="cpu", weights_only=False)
            array = np.asarray(stored)
            if array.ndim != 4:
                raise RuntimeError(f"RGB source shape is not 4D for {source}")
            if array.shape[-1] == 3:
                frames = array
            elif array.shape[1] == 3:
                frames = np.transpose(array, (0, 2, 3, 1))
            else:
                raise RuntimeError(f"RGB source channel axis is ambiguous for {source}")
            rounded = np.rint(frames)
            if not np.array_equal(frames, rounded):
                raise RuntimeError(f"RGB source is not integer-valued for {source}")
            if float(rounded.min()) < 0.0 or float(rounded.max()) > 255.0:
                raise RuntimeError(f"RGB source range differs for {source}")
            self._arrays[path] = rounded.astype(np.uint8, copy=False)
            self._hashes[path] = actual_sha256
        return self._arrays[path]

    @property
    def checked_source_hashes(self) -> dict[str, str]:
        return dict(self._hashes)


def classify_pair(anchor: np.ndarray, duplicate: np.ndarray) -> dict[str, Any]:
    if anchor.shape != duplicate.shape or anchor.ndim != 3 or anchor.shape[-1] != 3:
        raise RuntimeError("aligned RGB pair shapes differ")
    delta = np.abs(duplicate.astype(np.float32) - anchor.astype(np.float32))
    mean_absolute_delta = float(delta.mean())
    pixel_mean_delta = delta.mean(axis=-1)
    identical = bool(np.array_equal(anchor, duplicate))
    if identical:
        category = "identical_rgb"
    elif mean_absolute_delta < RGB_MOVING_THRESHOLD_UINT8:
        category = "near_static_rgb"
    else:
        category = "materially_changed_rgb"
    return {
        "category": category,
        "identical_array": identical,
        "mean_absolute_channel_delta_uint8": mean_absolute_delta,
        "maximum_pixel_mean_delta_uint8": float(pixel_mean_delta.max()),
        "moving_pixel_fraction_at_5_uint8": float(
            (pixel_mean_delta >= RGB_MOVING_THRESHOLD_UINT8).mean()
        ),
    }


def measure_environment(
    environment: str,
    source_root: Path,
    replacement_cache: Path,
    replacement_manifest: Path,
    final_receipt: Path,
    defective_manifest: Path,
    expected_repeated: int,
) -> dict[str, Any]:
    manifest = load_json(replacement_manifest)
    if manifest.get("environment") != environment:
        raise RuntimeError(f"{environment} replacement manifest environment differs")
    key_map = manifest_key_map(manifest, source_root)
    receipt = check_final_receipt(
        final_receipt,
        environment,
        replacement_manifest,
        defective_manifest,
        expected_repeated,
    )
    groups, payloads, repeated = duplicate_groups(replacement_cache)
    if payloads != 20000 or repeated != expected_repeated:
        raise RuntimeError(f"{environment} LMDB duplicate coverage differs")
    duplicate_group_list = [
        (payload_hash, keys) for payload_hash, keys in groups.items() if len(keys) > 1
    ]
    duplicate_group_list.sort(key=lambda item: item[1][0])

    source_cache = RGBSourceCache()
    occurrences: list[dict[str, Any]] = []
    for payload_hash, keys in duplicate_group_list:
        anchor_key = keys[0]
        if keys != sorted(keys):
            raise RuntimeError(f"{environment} duplicate group order is not deterministic")
        anchor_reference = key_map.get(anchor_key)
        if anchor_reference is None:
            raise RuntimeError(f"{environment} missing manifest alignment for {anchor_key}")
        anchor_source = source_cache.load(anchor_reference)
        for ordinal, duplicate_key in enumerate(keys[1:], start=1):
            duplicate_reference = key_map.get(duplicate_key)
            if duplicate_reference is None:
                raise RuntimeError(
                    f"{environment} missing manifest alignment for {duplicate_key}"
                )
            duplicate_source = source_cache.load(duplicate_reference)
            pair = classify_pair(
                anchor_source[anchor_reference["frame"]],
                duplicate_source[duplicate_reference["frame"]],
            )
            occurrences.append(
                {
                    "payload_sha256": payload_hash,
                    "group_size": len(keys),
                    "group_duplicate_ordinal": ordinal,
                    "anchor_key": anchor_key,
                    "duplicate_key": duplicate_key,
                    "anchor_source": anchor_reference,
                    "duplicate_source": duplicate_reference,
                    "rgb": pair,
                }
            )

    counts = defaultdict(int)
    for occurrence in occurrences:
        counts[occurrence["rgb"]["category"]] += 1
    if sum(counts.values()) != expected_repeated:
        raise RuntimeError(f"{environment} RGB category counts do not reconcile")
    pinned_path = source_root / "deformable" / environment / (
        "000997" if environment == "rope" else "000999"
    ) / "obses.pth"
    pinned_actual = sha256_file(pinned_path)
    if pinned_actual != PINNED_RGB_SHA256[environment]:
        raise RuntimeError(f"{environment} pinned RGB source hash differs")

    return {
        "environment": environment,
        "final_receipt": receipt,
        "replacement_manifest": {
            "path": str(replacement_manifest),
            "sha256": sha256_file(replacement_manifest),
            "manifest_id": manifest.get("manifest_id"),
            "data_mdb_sha256": manifest.get("data_mdb_sha256"),
            "producer_sha256": hashlib.sha256(
                json.dumps(
                    manifest["producer"],
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest(),
        },
        "defective_manifest": {
            "path": str(defective_manifest),
            "sha256": sha256_file(defective_manifest),
        },
        "duplicate_payloads": payloads,
        "duplicate_groups": len(duplicate_group_list),
        "duplicate_occurrences": expected_repeated,
        "rgb_category_counts": dict(
            sorted(
                {
                    "identical_rgb": counts["identical_rgb"],
                    "near_static_rgb": counts["near_static_rgb"],
                    "materially_changed_rgb": counts["materially_changed_rgb"],
                }.items()
            )
        ),
        "source_files_checked": len(source_cache.checked_source_hashes),
        "pinned_rgb_source": {"path": str(pinned_path), "sha256": pinned_actual},
        "occurrences": occurrences,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--replacement-root", type=Path, required=True)
    parser.add_argument("--rebuilt-root", type=Path, required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected_counts = {"rope": 997, "granular": 999}
    results = []
    for environment in ENVIRONMENTS:
        # The final admission receipts bind the immutable release-cache root,
        # whose entries are <environment>.lmdb (the producer staging cache has
        # an additional per-environment directory and is not receipt-bound).
        replacement_cache = args.replacement_root / f"{environment}.lmdb"
        replacement_manifest = replacement_cache / "manifest.json"
        defective_cache = args.rebuilt_root / f"{environment}.lmdb"
        defective_manifest = defective_cache / "manifest.json"
        final_receipt = args.final_root / environment / "admission.json"
        results.append(
            measure_environment(
                environment,
                args.source_root,
                replacement_cache,
                replacement_manifest,
                final_receipt,
                defective_manifest,
                expected_counts[environment],
            )
        )

    total = sum(result["duplicate_occurrences"] for result in results)
    materially_changed = sum(
        result["rgb_category_counts"]["materially_changed_rgb"] for result in results
    )
    if materially_changed:
        conclusion = (
            "duplicate corrected-depth payloads repeat despite materially changed "
            "aligned RGB in at least one complete environment measurement"
        )
    else:
        conclusion = (
            "all duplicate corrected-depth payloads are explained by identical or "
            "near-static aligned RGB under the frozen rule"
        )
    output = {
        "schema": "dinocular.rg-duplicate-rgb-measurement.v1",
        "integrated_route_commit": "44ac52f44c64cd471dc773e05584b6394304822a",
        "inputs_unchanged": True,
        "similarity_rule": {
            "pair": "lexicographically first key in each equal serialized-payload hash group versus each excess member",
            "rgb_source": "manifest-bound raw integer-valued RGB source frame; no future search or substitution",
            "identical": "exact RGB array equality",
            "near_static": "non-identical and mean absolute per-channel RGB delta < 5.0 uint8",
            "materially_changed": "mean absolute per-channel RGB delta >= 5.0 uint8",
            "threshold_uint8": RGB_MOVING_THRESHOLD_UINT8,
            "category_order": [
                "identical_rgb",
                "near_static_rgb",
                "materially_changed_rgb",
            ],
        },
        "expected_reconciliation": {
            "rope": 997,
            "granular": 999,
            "total": 1996,
        },
        "observed_reconciliation": {
            "rope": results[0]["duplicate_occurrences"],
            "granular": results[1]["duplicate_occurrences"],
            "total": total,
            "coverage_pass": total == 1996
            and all(
                result["duplicate_occurrences"]
                == {"rope": 997, "granular": 999}[result["environment"]]
                for result in results
            ),
        },
        "conclusion": conclusion,
        "environments": results,
        "checkpoint_sha256": CHECKPOINT_SHA256,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256_file(args.output),
                "counts": {
                    result["environment"]: result["rgb_category_counts"]
                    for result in results
                },
                "coverage": output["observed_reconciliation"],
                "conclusion": conclusion,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
