#!/usr/bin/env python3
"""Materialize new-identity corrected Rope/Granular seed-one training cards."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ENVIRONMENTS = ("rope", "granular")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--admission-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    release = json.loads(args.release.read_text())
    if release.get("state") != "READY_FOR_END_TO_END_VALIDATION":
        raise RuntimeError("replacement release identity is not ready")
    contract_path = Path(release["contract"]["path"])
    contract_sha256 = release["contract"]["sha256"]
    producer_sha256 = release["producer_sha256"]
    records = {}

    old_root = (
        args.project
        / "outputs/campaign-seed1/native-seed1-20260728c"
    )
    cards_root = args.output_root / "cards"
    cards_root.mkdir(parents=True, exist_ok=False)
    for environment in ENVIRONMENTS:
        admission_path = args.admission_root / environment / "admission.json"
        admission = json.loads(admission_path.read_text())
        if admission.get("state") != "PASS":
            raise RuntimeError(f"{environment} admission receipt is not PASS")
        cache = Path(release["environments"][environment]["cache"])
        manifest_path = cache / "manifest.json"
        manifest_sha256 = release["environments"][environment][
            "manifest_sha256"
        ]
        reader_validation = release["environments"][environment][
            "reader_validation"
        ]
        old_card_path = old_root / "cards" / f"p3-{environment}-dinocular-s1.yaml"
        old_card = yaml.safe_load(old_card_path.read_text())
        old_base = old_root / environment / "dinocular"
        new_base = args.output_root / environment / "dinocular"
        new_run = new_base / "run"
        new_card_path = cards_root / f"p3-{environment}-dinocular-mapanything-repair-s1.yaml"

        card = copy.deepcopy(old_card)
        card["run_id"] = f"p3-{environment}-dinocular-mapanything-repair-s1"
        card["run_dir"] = str(new_run)
        card["artifacts"]["native_depth_contract"] = {
            "path": str(contract_path),
            "sha256": contract_sha256,
        }
        card["artifacts"].pop("native_depth_contract_index", None)
        card["artifacts"].pop("native_depth_independent_acceptance", None)
        card["artifacts"]["depth_admission_validation"] = {
            "path": str(admission_path),
            "sha256": sha256_file(admission_path),
            "receipt_sha256": admission["receipt_sha256"],
        }
        card["environment_variables"].update(
            {
                "DINOCULAR_NATIVE_DEPTH_CONTRACT": str(contract_path),
                "DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256": contract_sha256,
                "DINOCULAR_CACHE_PRODUCER_SHA256": producer_sha256,
            }
        )
        card["depth_inputs"] = {
            "environment": environment,
            "producer": "mapanything_recovered_framewise_raw_depth_z",
            "producer_sha256": producer_sha256,
            "cache_dir": str(cache),
            "cache_manifest_sha256": manifest_sha256,
            "validation_path": reader_validation["path"],
            "validation_sha256": reader_validation["sha256"],
            "native_contract_path": str(contract_path),
            "native_contract_sha256": contract_sha256,
            "checkpoint_sha256": card["artifacts"]["dinocular_student"]["sha256"],
            "quantity": "later_pinned_MapAnything_depth_z_proxy",
            "units": "declared_metric_Z_depth_nonphysical_later_checkpoint_proxy",
            "normalization": "none_raw_depth_z",
            "invalid_policy": (
                "reject_nonfinite_or_negative_preserve_upstream_masked_exact_zero"
            ),
        }
        replacements = {
            str(old_card["depth_inputs"]["cache_dir"]): str(cache),
            old_card["depth_inputs"]["cache_manifest_sha256"]: manifest_sha256,
            str(old_card["depth_inputs"]["validation_path"]): reader_validation[
                "path"
            ],
            old_card["depth_inputs"]["validation_sha256"]: reader_validation[
                "sha256"
            ],
            str(old_card["depth_inputs"]["native_contract_path"]): str(
                contract_path
            ),
            old_card["depth_inputs"]["native_contract_sha256"]: contract_sha256,
            old_card["depth_inputs"]["producer_sha256"]: producer_sha256,
        }
        card["overrides"] = [
            next(
                (
                    value.replace(old, new)
                    for old, new in replacements.items()
                    if old in value
                ),
                value,
            )
            for value in card["overrides"]
        ]
        card["config_sha256"] = hashlib.sha256(
            canonical_bytes(card["overrides"])
        ).hexdigest()
        card["producer_decision"] = {
            "path": str(admission_path),
            "sha256": sha256_file(admission_path),
            "winner": "mapanything_recovered_framewise_raw_depth_z",
        }
        card["assumption_tags"] = [
            "recovered_mapanything_family_later_pinned_checkpoint_proxy"
        ]
        card["recovery_disclosure"] = {
            "original_checkpoint_snapshot": "unavailable",
            "original_invocation": "unavailable",
            "compatibility_proven": (
                "producer_family_and_raw_depth_z_no_normalization_input_path"
            ),
            "non_equivalence": (
                "later pinned MapAnything output is not claimed byte-equivalent "
                "to the unavailable original ImageNet depth archive"
            ),
        }
        card["depth_repair_admission"] = {
            "path": str(admission_path),
            "sha256": sha256_file(admission_path),
            "receipt_sha256": admission["receipt_sha256"],
        }
        card.pop("native_depth_acceptance", None)
        card.pop("run_card_sha256", None)
        card["run_card_sha256"] = hashlib.sha256(canonical_bytes(card)).hexdigest()
        new_card_path.write_text(yaml.safe_dump(card, sort_keys=False))
        card_file_sha256 = sha256_file(new_card_path)

        old_wrapper_path = old_base / "direct_segment.sbatch"
        wrapper = old_wrapper_path.read_text()
        wrapper_replacements = {
            str(old_base): str(new_base),
            str(old_card_path): str(new_card_path),
            old_card["run_card_sha256"]: card["run_card_sha256"],
            "nat-" + environment + "-dino-s1": (
                "rgfix-" + environment + "-dino-s1"
            ),
            **replacements,
        }
        for old, new in wrapper_replacements.items():
            wrapper = wrapper.replace(old, new)
        new_base.mkdir(parents=True, exist_ok=False)
        wrapper_path = new_base / "direct_segment.sbatch"
        wrapper_path.write_text(wrapper)
        wrapper_path.chmod(0o755)
        records[environment] = {
            "card": str(new_card_path),
            "card_file_sha256": card_file_sha256,
            "card_semantic_sha256": card["run_card_sha256"],
            "wrapper": str(wrapper_path),
            "wrapper_sha256": sha256_file(wrapper_path),
            "run_dir": str(new_run),
        }
    receipt = {
        "schema": "dinocular.rg-corrected-seed1-materialization.v1",
        "state": "READY",
        "release_sha256": sha256_file(args.release),
        "environments": records,
    }
    receipt_path = args.output_root / "materialization.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
