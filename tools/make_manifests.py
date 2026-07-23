#!/usr/bin/env python3
"""Materialize locked P2, P2a, P3, and P4 immutable YAML run cards."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness_common import (  # noqa: E402
    HarnessError,
    LOCKED_ARMS,
    LOCKED_ENVS,
    LOCKED_FRAMESKIPS,
    LOCKED_HORIZONS,
    LOCKED_SEEDS,
    LOCKED_TARGETS,
    LEGACY_RUN_CARD_SCHEMA,
    RUN_CARD_SCHEMA,
    RECOVERED_CONTRACT_ASSUMPTION,
    canonical_json_bytes,
    depth_inputs,
    derive_segment_sizing,
    finalize_run_card,
    load_contract_index,
    load_native_contract_index,
    legacy_depth_inputs,
    load_json,
    load_matrix,
    load_yaml,
    require_launch_authorization,
    require_real_marvin_path,
    require_regular_file_no_alias,
    resolve_authorization_prerequisites,
    sha256_bytes,
    sha256_file,
    source_evidence,
    validate_spec,
    write_matrix,
)


def _parse_csv(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result or len(set(result)) != len(result):
        raise HarnessError(f"invalid comma-separated value: {value!r}")
    return result


def _parse_int_map(value: str) -> dict[str, int]:
    result = {}
    for item in _parse_csv(value):
        key, separator, raw = item.partition("=")
        if not separator:
            raise HarnessError(f"invalid environment map item: {item!r}")
        result[key] = int(raw)
    return result


def _resolve_inputs(
    args: argparse.Namespace,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    spec_path = require_regular_file_no_alias(
        Path(args.spec),
        "materialization study specification",
    )
    loaded_spec = load_yaml(spec_path)
    validate_spec(loaded_spec)
    spec = copy.deepcopy(dict(loaded_spec))
    contract_path = require_regular_file_no_alias(
        Path(args.contracts_index or str(spec["contracts_index"])),
        "materialization contracts index",
    )
    raw_index = load_yaml(contract_path)
    entries = raw_index.get("entries")
    release_record = raw_index.get("empirical_runtime_release")
    if not isinstance(entries, Mapping) or not isinstance(release_record, Mapping):
        raise HarnessError("materialization contracts index lacks empirical bindings")
    empirical_entry = entries.get("pusht/mapanything_recovered_framewise")
    if not isinstance(empirical_entry, Mapping):
        raise HarnessError("materialization empirical PushT entry is absent")
    evidence = source_evidence(spec, args.local_code_root)
    bindings = resolve_authorization_prerequisites(
        spec,
        contracts_index_sha256=sha256_file(contract_path),
        empirical_contract_sha256=str(empirical_entry.get("contract_sha256")),
        empirical_runtime_release_sha256=str(release_record.get("release_sha256")),
        source_commit=str(evidence["source_commit"]),
        source_file_sha256=evidence["source_file_sha256"],
    )
    authorization_path = getattr(args, "launch_authorization", None)
    authorization_sha = getattr(args, "launch_authorization_sha256", None)
    spec["launch_authorization"] = {
        "status": "AUTHORIZED",
        "path": str(authorization_path) if authorization_path is not None else None,
        "sha256": authorization_sha,
    }
    spec["authorization_bindings"] = bindings
    require_launch_authorization(spec, operation="materialize")
    contracts = load_contract_index(contract_path)
    return spec, contracts, evidence


def _resolve_native_inputs(
    args: argparse.Namespace,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Resolve unchanged native-v1 inputs without empirical authority surfaces."""

    spec_path = require_regular_file_no_alias(
        Path(args.spec),
        "materialization study specification",
    )
    loaded_spec = load_yaml(spec_path)
    validate_spec(loaded_spec)
    spec = copy.deepcopy(dict(loaded_spec))
    contract_path = require_regular_file_no_alias(
        Path(args.contracts_index or str(spec["contracts_index"])),
        "materialization contracts index",
    )
    contracts = load_native_contract_index(contract_path)
    evidence = source_evidence(spec, args.local_code_root)
    return spec, contracts, evidence


def _base_card(
    spec: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    kind: str,
    run_id: str,
    environment: str,
    arm: str,
    seed: int,
    schema: str = RUN_CARD_SCHEMA,
) -> dict[str, Any]:
    if environment not in LOCKED_ENVS or arm not in LOCKED_ARMS:
        raise HarnessError("run card requests an unlocked environment or arm")
    env_record = spec["environments"][environment]
    run_dir = f"{spec['study_root']}/outputs/{kind}/{run_id}"
    card = {
        "schema": schema,
        "kind": kind,
        "run_id": run_id,
        "environment": environment,
        "arm": arm,
        "seed": int(seed),
        "run_dir": run_dir,
        "code_root": spec["code_root"],
        "source_commit": evidence["source_commit"],
        "source_file_sha256": copy.deepcopy(evidence["source_file_sha256"]),
        "artifacts": copy.deepcopy(evidence["artifacts"]),
        "container": copy.deepcopy(evidence["container"]),
        "target_steps": int(env_record["target_steps"]),
        "frameskip": int(env_record["frameskip"]),
        "horizons": list(env_record["horizons"]),
        "batch_size": 32,
        "predictor_lr": 0.00005,
        "decoder": False,
        "strict_resume": True,
        "depends_on": [],
        "environment_variables": {
            "DINOV2_REPO": f"{spec['study_root']}/code/dinov2",
            "DINOV2_VITS14_WEIGHTS": spec["artifacts"]["dinov2"]["path"],
        },
    }
    if schema == RUN_CARD_SCHEMA:
        card.update(
            {
                "launch_authorization_subject": spec.get(
                    "launch_authorization_subject"
                ),
                "launch_authorization": copy.deepcopy(
                    spec.get("launch_authorization")
                ),
                "authorization_bindings": copy.deepcopy(
                    spec.get("authorization_bindings", {})
                ),
            }
        )
    elif schema != LEGACY_RUN_CARD_SCHEMA:
        raise HarnessError("unsupported materialized run-card schema")
    return card


def _depth_overrides(card: dict[str, Any], inputs: Mapping[str, Any]) -> None:
    values = copy.deepcopy(dict(inputs))
    if values.get("contract_kind") == "empirical_lossy_cache":
        values["adapter_mode"] = (
            "exact_constant_zero_numeric"
            if card["arm"] == "dinocular_zerodepth"
            else "proxy_depth_z"
        )
        runtime_paths = values.get("runtime_paths")
        if not isinstance(runtime_paths, Mapping):
            raise HarnessError("empirical runtime paths are unavailable")
        card["assumption_tags"] = [RECOVERED_CONTRACT_ASSUMPTION]
        card["environment_variables"].update(
            {
                "DINOCULAR_STUDENT_WEIGHTS": runtime_paths["checkpoint"],
                "DINOCULAR_EMPIRICAL_DEPTH_CONTRACT": runtime_paths[
                    "empirical_contract"
                ],
                "DINOCULAR_EMPIRICAL_DEPTH_CONTRACT_SHA256": values["empirical_contract_sha256"],
                "DINOCULAR_EMPIRICAL_RUNTIME_RELEASE": runtime_paths[
                    "empirical_runtime_release"
                ],
                "DINOCULAR_EMPIRICAL_RUNTIME_RELEASE_SHA256": values["empirical_runtime_release_sha256"],
                "DINOCULAR_DEPTH_INPUT_MODE": "empirical_lossy_cache_v1",
                "DINOCULAR_EMPIRICAL_ADAPTER_ID": values["adapter_id"],
                "DINOCULAR_EMPIRICAL_ADAPTER_MODE": values["adapter_mode"],
                "DINOCULAR_EMPIRICAL_ZERO_INTERVENTION": (
                    "true" if card["arm"] == "dinocular_zerodepth" else "false"
                ),
            }
        )
    else:
        card["environment_variables"].update(
            {
                "DINOCULAR_STUDENT_WEIGHTS": card["artifacts"]["dinocular_student"]["path"],
                "DINOCULAR_NATIVE_DEPTH_CONTRACT": values["native_contract_path"],
                "DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256": values["native_contract_sha256"],
                "DINOCULAR_CACHE_PRODUCER_SHA256": values["producer_sha256"],
            }
        )
    card["depth_inputs"] = values


def _training_overrides(spec: Mapping[str, Any], card: dict[str, Any]) -> list[str]:
    environment = str(card["environment"])
    arm = str(card["arm"])
    env_record = spec["environments"][environment]
    inputs = card.get("depth_inputs")
    encoder_config = arm
    if isinstance(inputs, Mapping) and inputs.get("contract_kind") == "empirical_lossy_cache":
        encoder_config = (
            "dinocular_zerodepth_pusht_empirical"
            if arm == "dinocular_zerodepth"
            else "dinocular_pusht_empirical"
        )
    overrides = [
        f"env={env_record['hydra_env']}",
        f"encoder={encoder_config}",
        f"training.seed={card['seed']}",
        "training.predictor_lr=5e-5",
        "training.strict_determinism=true",
        "training.resume_from=auto",
        "training.batch_size=32",
        "env.num_workers=0",
        f"frameskip={card['frameskip']}",
        f"num_hist={env_record['num_hist']}",
        "num_pred=1",
        "has_decoder=false",
        "model.train_encoder=false",
        "model.train_predictor=true",
        "model.train_decoder=false",
        "plan_settings.plan_cfg_path=null",
    ]
    if env_record.get("object_name") is not None:
        overrides.append(f"env.dataset.object_name={env_record['object_name']}")
        overrides.append(f"env.kwargs.object_name={env_record['object_name']}")
    if inputs is not None:
        if inputs.get("contract_kind") == "empirical_lossy_cache":
            overrides.extend(
                [
                    "+env.dataset.depth_contract_kind=empirical_lossy_cache",
                    f"+env.dataset.empirical_depth_contract_path={inputs['empirical_contract_path']}",
                    f"+env.dataset.empirical_depth_contract_sha256={inputs['empirical_contract_sha256']}",
                    f"+env.dataset.empirical_runtime_release_path={inputs['empirical_runtime_release_path']}",
                    f"+env.dataset.empirical_runtime_release_sha256={inputs['empirical_runtime_release_sha256']}",
                    f"+env.dataset.depth_checkpoint_sha256={inputs['checkpoint_sha256']}",
                ]
            )
        else:
            overrides.extend(
                [
                    f"+env.dataset.depth_cache_dir={inputs['cache_dir']}",
                    f"+env.dataset.depth_cache_manifest_sha256={inputs['cache_manifest_sha256']}",
                    f"+env.dataset.depth_validation_path={inputs['validation_path']}",
                    f"+env.dataset.depth_validation_sha256={inputs['validation_sha256']}",
                    f"+env.dataset.native_depth_contract_path={inputs['native_contract_path']}",
                    f"+env.dataset.native_depth_contract_sha256={inputs['native_contract_sha256']}",
                    f"+env.dataset.depth_cache_producer_sha256={inputs['producer_sha256']}",
                    f"+env.dataset.depth_checkpoint_sha256={inputs['checkpoint_sha256']}",
                ]
            )
    heldout = card.get("heldout_loss_manifest")
    if heldout is not None:
        overrides.extend(
            [
                "training.p3_completion_enabled=true",
                f"training.p3_heldout_manifest={heldout['path']}",
                f"training.p3_heldout_manifest_sha256={heldout['sha256']}",
                f"training.p3_heldout_metadata={heldout['metadata_path']}",
                f"training.p3_heldout_metadata_sha256={heldout['metadata_sha256']}",
                f"training.p3_data_manifest_sha256={heldout['data_manifest_sha256']}",
                f"training.p3_split_sha256={heldout['split_sha256']}",
            ]
        )
    if card.get("gate_mode") == "timing":
        timing = card["timing"]
        overrides.extend(
            [
                "training.resume_from=null",
                "training.checkpoint_every_steps=0",
                f"training.timing_output={card['run_dir']}/timing_result.json",
                f"training.timing_run_card={card['run_dir']}/timing_runtime_card.yaml",
                f"training.timing_warmup_steps={timing['warmup_steps']}",
                f"training.timing_measured_steps={timing['fixed_steps']}",
                f"training.timing_projection_target_steps={LOCKED_TARGETS[environment]}",
            ]
        )
    card["config_sha256"] = sha256_bytes(canonical_json_bytes(overrides))
    return overrides


def _finish_card(spec: Mapping[str, Any], card: dict[str, Any]) -> Mapping[str, Any]:
    card["overrides"] = _training_overrides(spec, card)
    return finalize_run_card(card)


def _segment_record(
    spec: Mapping[str, Any],
    evidence: Mapping[str, Any],
    rates: Path,
    *,
    arm: str,
    environment: str,
    target_steps: int,
) -> Mapping[str, Any]:
    record = derive_segment_sizing(
        summary_path=rates,
        arm=arm,
        environment=environment,
        target_steps=target_steps,
        policy=spec["segment_sizing"],
    )
    if record.get("timing_source_commit") != evidence["source_commit"]:
        raise HarnessError(
            "timing summary and materialized run cards use different commits"
        )
    return record


def make_p2(args: argparse.Namespace) -> Mapping[str, Any]:
    spec, contracts, evidence = _resolve_native_inputs(args)
    kind = args.kind
    producer = args.producer
    if producer not in spec["p2a"]["producers"]:
        raise HarnessError("P2 producer must be one of the two predeclared candidates")
    cards = []
    for arm in LOCKED_ARMS:
        for environment in LOCKED_ENVS:
            run_id = f"p2-{kind}-{arm}-{environment}-s1"
            card = _base_card(
                spec,
                evidence,
                kind=f"p2-{kind}",
                run_id=run_id,
                environment=environment,
                arm=arm,
                seed=1,
                schema=LEGACY_RUN_CARD_SCHEMA,
            )
            if arm != "dino_pinned":
                _depth_overrides(card, legacy_depth_inputs(contracts, producer, environment))
            if kind == "geometry":
                card["gate_mode"] = "geometry"
                card["target_steps"] = 1
                card["segment_steps"] = 1
            else:
                card["gate_mode"] = "timing"
                card["depends_on"] = [f"p2-geometry-{arm}-{environment}-s1"]
                card["target_steps"] = int(spec["p2"]["fixed_steps"]) + int(
                    spec["p2"]["warmup_steps"]
                )
                card["segment_steps"] = card["target_steps"]
                card["timing"] = {
                    "fixed_steps": int(spec["p2"]["fixed_steps"]),
                    "warmup_steps": int(spec["p2"]["warmup_steps"]),
                }
            cards.append(_finish_card(spec, card))
    if len(cards) != 12:
        raise HarnessError("P2 materialization did not produce exactly 12 cards")
    return write_matrix(
        args.out,
        kind=f"p2-{kind}",
        cards=cards,
        source_commit=evidence["source_commit"],
    )


def make_producer_pilot(args: argparse.Namespace) -> Mapping[str, Any]:
    spec, contracts, evidence = _resolve_native_inputs(args)
    if args.env != "pusht" or args.seed != 1 or args.target_steps != 123858:
        raise HarnessError("P2a must be PushT seed 1 at exactly 123858 steps")
    producers = _parse_csv(args.producers)
    if producers != spec["p2a"]["producers"]:
        raise HarnessError("P2a must contain exactly the two locked producers in order")
    cards = []
    for producer in producers:
        run_id = f"p2a-pusht-dinocular-s1-{producer}"
        card = _base_card(
            spec,
            evidence,
            kind="p2a-producer-pilot",
            run_id=run_id,
            environment="pusht",
            arm="dinocular",
            seed=1,
            schema=LEGACY_RUN_CARD_SCHEMA,
        )
        _depth_overrides(card, legacy_depth_inputs(contracts, producer, "pusht"))
        card["segment_sizing"] = _segment_record(
            spec,
            evidence,
            args.rates,
            arm="dinocular",
            environment="pusht",
            target_steps=card["target_steps"],
        )
        card["segment_steps"] = card["segment_sizing"]["derived_segment_steps"]
        card["producer_pilot"] = {
            "producer": producer,
            "decision_horizons": [5, 10],
            "paired_manifest_required": True,
        }
        card["assumption_tags"] = (
            [RECOVERED_CONTRACT_ASSUMPTION]
            if producer == "mapanything_recovered_framewise"
            else []
        )
        card["depends_on"] = [
            f"p2-timing-{arm}-{environment}-s1"
            for arm in LOCKED_ARMS
            for environment in LOCKED_ENVS
        ]
        cards.append(_finish_card(spec, card))
    if len(cards) != 2:
        raise HarnessError("P2a materialization did not produce exactly two cards")
    return write_matrix(
        args.out,
        kind="p2a-producer-pilot",
        cards=cards,
        source_commit=evidence["source_commit"],
    )


def _load_winner(path: Path, spec: Mapping[str, Any]) -> str:
    path = require_regular_file_no_alias(path, "P2a producer decision")
    decision = load_json(path)
    if (
        decision.get("schema") != "dino-wm-p2a-producer-decision-v1"
        or decision.get("status") != "PASS"
        or decision.get("tie_tolerance") != spec["p2a"]["tie_tolerance"]
    ):
        raise HarnessError("P2a producer decision is absent, blocked, or incompatible")
    winner = decision.get("winner")
    if winner not in spec["p2a"]["producers"]:
        raise HarnessError("P2a producer decision names an unregistered winner")
    expected_assumptions = (
        [RECOVERED_CONTRACT_ASSUMPTION]
        if winner == "mapanything_recovered_framewise"
        else []
    )
    if decision.get("winner_assumption_tags") != expected_assumptions:
        raise HarnessError("P2a producer decision assumption provenance differs")
    return str(winner)


def _heldout_manifest_record(
    directory: Path,
    environment: str,
    source_commit: str,
    target_steps: int,
) -> Mapping[str, Any]:
    path = require_regular_file_no_alias(
        directory / f"heldout_{environment}.jsonl",
        f"{environment} held-out manifest",
    )
    metadata_path = require_regular_file_no_alias(
        path.with_suffix(".meta.json"), f"{environment} held-out metadata"
    )
    metadata = load_json(metadata_path)
    try:
        rows = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(
            f"P3 held-out manifest is invalid for {environment}"
        ) from exc
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        raise HarnessError(f"P3 held-out manifest is invalid for {environment}")
    if (
        metadata.get("schema") != "dino-wm.p3-heldout-manifest.v1"
        or metadata.get("environment") != environment
        or metadata.get("selection") != "all_validation_examples"
        or metadata.get("source_commit") != source_commit
        or not isinstance(metadata.get("target_steps"), int)
        or isinstance(metadata.get("target_steps"), bool)
        or metadata.get("target_steps") != target_steps
        or metadata.get("rounding_rule") != "ceil(target_steps*percent/100)"
        or metadata.get("manifest_sha256") != sha256_file(path)
        or not isinstance(metadata.get("entry_count"), int)
        or isinstance(metadata.get("entry_count"), bool)
        or metadata.get("entry_count", 0) <= 0
        or metadata.get("entry_count") != len(rows)
        or len(str(metadata.get("data_manifest_sha256"))) != 64
        or len(str(metadata.get("split_sha256"))) != 64
    ):
        raise HarnessError(f"P3 held-out manifest contract differs for {environment}")
    return {
        "path": require_real_marvin_path(str(path), f"{environment} held-out manifest"),
        "sha256": sha256_file(path),
        "metadata_path": require_real_marvin_path(
            str(metadata_path), f"{environment} held-out metadata"
        ),
        "metadata_sha256": sha256_file(metadata_path),
        "data_manifest_sha256": metadata["data_manifest_sha256"],
        "split_sha256": metadata["split_sha256"],
        "selection": "all_validation_examples",
        "entry_count": metadata["entry_count"],
        "target_steps": metadata["target_steps"],
        "rounding_rule": metadata["rounding_rule"],
    }


def materialize_training_card(
    spec: Mapping[str, Any],
    contracts: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    winner: str,
    heldout: Mapping[str, Mapping[str, Any]],
    rates: Path,
    producer_decision: Path,
    arm: str,
    environment: str,
    seed: int,
) -> Mapping[str, Any]:
    """Materialize one exact P3 card; the matrix entrypoint locks the full grid."""

    run_id = f"p3-{environment}-{arm}-s{seed}"
    card = _base_card(
        spec,
        evidence,
        kind="p3-training",
        run_id=run_id,
        environment=environment,
        arm=arm,
        seed=seed,
    )
    if arm != "dino_pinned":
        _depth_overrides(card, depth_inputs(contracts, winner, environment))
    card["heldout_loss_manifest"] = copy.deepcopy(heldout[environment])
    card["initialization_policy"] = {
        "predictor": "fresh_seeded",
        "action_encoder": "fresh_seeded",
        "proprio_encoder": "fresh_seeded",
        "seed": seed,
        "encoder": "frozen",
    }
    card["optimizer_policy"] = {
        "predictor": "adamw",
        "predictor_lr": 0.00005,
        "action_proprio": "adamw",
        "action_proprio_lr": 0.0005,
    }
    card["schedule_policy"] = "fixed_learning_rates"
    if arm == "dino_pinned":
        card["encoder_boundary"] = "not_applicable"
    elif card["depth_inputs"].get("contract_kind") == "empirical_lossy_cache":
        card["encoder_boundary"] = (
            "exact_constant_zero_numeric_and_audit_mask"
            if arm == "dinocular_zerodepth"
            else "empirical_proxy_depth_and_payload_presence_mask"
        )
    else:
        card["encoder_boundary"] = (
            "manifest_neutral_depth_and_mask"
            if arm == "dinocular_zerodepth"
            else "informative_depth_and_mask"
        )
    card["segment_sizing"] = _segment_record(
        spec,
        evidence,
        rates,
        arm=arm,
        environment=environment,
        target_steps=card["target_steps"],
    )
    card["segment_steps"] = card["segment_sizing"]["derived_segment_steps"]
    producer_decision = require_regular_file_no_alias(
        producer_decision, "P3 producer decision"
    )
    card["producer_decision"] = {
        "path": str(producer_decision),
        "sha256": sha256_file(producer_decision),
        "winner": winner,
    }
    card["assumption_tags"] = (
        [RECOVERED_CONTRACT_ASSUMPTION]
        if winner == "mapanything_recovered_framewise" and arm != "dino_pinned"
        else []
    )
    return _finish_card(spec, card)


def make_training(args: argparse.Namespace) -> Mapping[str, Any]:
    spec, contracts, evidence = _resolve_inputs(args)
    if tuple(_parse_csv(args.encoders)) != LOCKED_ARMS:
        raise HarnessError("training matrix must contain exactly three locked arms")
    if tuple(_parse_csv(args.envs)) != LOCKED_ENVS:
        raise HarnessError(
            "training matrix must contain exactly four locked environments"
        )
    if tuple(int(value) for value in _parse_csv(args.seeds)) != LOCKED_SEEDS:
        raise HarnessError("training matrix must contain exactly seeds 1,2,3")
    if args.batch_size != 32 or args.predictor_lr != 0.00005 or args.decoder != "off":
        raise HarnessError(
            "training batch, LR, or decoder differs from the locked protocol"
        )
    if _parse_int_map(args.target_steps) != LOCKED_TARGETS:
        raise HarnessError("training target-step map differs from the locked protocol")
    if _parse_int_map(args.frameskips) != LOCKED_FRAMESKIPS:
        raise HarnessError("training frameskip map differs from the locked protocol")
    winner = _load_winner(args.producer_decision, spec)
    heldout = {
        environment: _heldout_manifest_record(
            args.heldout_manifests_dir,
            environment,
            evidence["source_commit"],
            LOCKED_TARGETS[environment],
        )
        for environment in LOCKED_ENVS
    }
    for environment in LOCKED_ENVS:
        depth_inputs(contracts, winner, environment)

    cards = []
    for arm in LOCKED_ARMS:
        for environment in LOCKED_ENVS:
            for seed in LOCKED_SEEDS:
                cards.append(
                    materialize_training_card(
                        spec,
                        contracts,
                        evidence,
                        winner=winner,
                        heldout=heldout,
                        rates=args.rates,
                        producer_decision=args.producer_decision,
                        arm=arm,
                        environment=environment,
                        seed=seed,
                    )
                )
    if len(cards) != 36:
        raise HarnessError("P3 materialization did not produce exactly 36 cards")
    return write_matrix(
        args.out,
        kind="p3-training",
        cards=cards,
        source_commit=evidence["source_commit"],
    )


def _fixed_manifest_record(directory: Path, environment: str) -> Mapping[str, Any]:
    path = require_regular_file_no_alias(
        directory / f"openloop_{environment}.jsonl",
        f"{environment} fixed evaluation manifest",
    )
    meta_path = require_regular_file_no_alias(
        path.with_suffix(".meta.json"),
        f"{environment} fixed evaluation manifest metadata",
    )
    metadata = load_json(meta_path)
    if (
        metadata.get("environment") != environment
        or metadata.get("frameskip") != LOCKED_FRAMESKIPS[environment]
        or metadata.get("horizons") != LOCKED_HORIZONS[environment]
    ):
        raise HarnessError(
            f"fixed evaluation manifest contract differs for {environment}"
        )
    return {
        "path": require_real_marvin_path(str(path), f"{environment} manifest"),
        "sha256": sha256_file(path),
        "metadata_path": require_real_marvin_path(
            str(meta_path), f"{environment} manifest metadata"
        ),
        "metadata_sha256": sha256_file(meta_path),
    }


def make_open_loop(args: argparse.Namespace) -> Mapping[str, Any]:
    spec, _contracts, evidence = _resolve_inputs(args)
    training, training_cards = load_matrix(args.training_matrix)
    if training.get("kind") != "p3-training" or training.get("card_count") != 36:
        raise HarnessError("P4 requires the accepted immutable 36-cell P3 matrix")
    if training.get("source_commit") != evidence["source_commit"]:
        raise HarnessError("P3 and P4 source commits differ")
    p3_by_id = {str(card["run_id"]): card for card in training_cards}
    p3_refs = {str(record["run_id"]): record for record in training["cards"]}
    manifest_records = {
        environment: _fixed_manifest_record(args.manifests_dir, environment)
        for environment in LOCKED_ENVS
    }
    cards = []
    for arm in LOCKED_ARMS:
        for environment in LOCKED_ENVS:
            for seed in LOCKED_SEEDS:
                run_id = f"p4-{environment}-{arm}-s{seed}"
                training_run_id = f"p3-{environment}-{arm}-s{seed}"
                p3_card = p3_by_id.get(training_run_id)
                p3_reference = p3_refs.get(training_run_id)
                if p3_card is None or p3_reference is None:
                    raise HarnessError(
                        f"P4 card has no exact hashed P3 card: {training_run_id}"
                    )
                card = _base_card(
                    spec,
                    evidence,
                    kind="p4-open-loop",
                    run_id=run_id,
                    environment=environment,
                    arm=arm,
                    seed=seed,
                    schema=str(p3_card["schema"]),
                )
                card["fixed_manifest"] = manifest_records[environment]
                card["training_run_id"] = training_run_id
                card["training_run_dir"] = p3_card["run_dir"]
                card["segment_steps"] = p3_card["segment_steps"]
                card["segment_sizing"] = copy.deepcopy(p3_card["segment_sizing"])
                card["training_run_card"] = {
                    "path": p3_reference["path"],
                    "file_sha256": p3_reference["file_sha256"],
                    "run_card_sha256": p3_reference["run_card_sha256"],
                }
                card["heldout_loss_manifest"] = copy.deepcopy(
                    p3_card["heldout_loss_manifest"]
                )
                card["initialization_policy"] = copy.deepcopy(
                    p3_card["initialization_policy"]
                )
                card["optimizer_policy"] = copy.deepcopy(p3_card["optimizer_policy"])
                card["schedule_policy"] = p3_card["schedule_policy"]
                card["encoder_boundary"] = p3_card["encoder_boundary"]
                card["training_completion_receipt"] = {
                    "path": f"{p3_card['run_dir']}/final_acceptance.json",
                    "schema": "dino-wm.p3-final-acceptance.v1",
                    "training_run_card_sha256": p3_card["run_card_sha256"],
                }
                if p3_card.get("depth_inputs") is not None:
                    _depth_overrides(card, p3_card["depth_inputs"])
                card["environment_variables"] = copy.deepcopy(
                    p3_card["environment_variables"]
                )
                card["assumption_tags"] = copy.deepcopy(
                    p3_card.get("assumption_tags", [])
                )
                card["depends_on"] = [card["training_run_id"]]
                finished = _finish_card(spec, card)
                if (
                    finished["overrides"] != p3_card["overrides"]
                    or finished["config_sha256"] != p3_card["config_sha256"]
                ):
                    raise HarnessError(
                        f"P4/P3 resolved training configuration differs for {card['training_run_id']}"
                    )
                cards.append(finished)
    if len(cards) != 36:
        raise HarnessError("P4 materialization did not produce exactly 36 cards")
    return write_matrix(
        args.out,
        kind="p4-open-loop",
        cards=cards,
        source_commit=evidence["source_commit"],
    )


def make_producer_pilot_eval(args: argparse.Namespace) -> Mapping[str, Any]:
    spec, _contracts, evidence = _resolve_native_inputs(args)
    training, training_cards = load_matrix(args.training_matrix)
    if (
        training.get("kind") != "p2a-producer-pilot"
        or training.get("card_count") != 2
        or training.get("source_commit") != evidence["source_commit"]
    ):
        raise HarnessError("P2a evaluation requires the exact two-card pilot matrix")
    by_id = {str(card["run_id"]): card for card in training_cards}
    references = {str(item["run_id"]): item for item in training["cards"]}
    fixed_manifest = _fixed_manifest_record(args.manifests_dir, "pusht")
    cards = []
    for producer in spec["p2a"]["producers"]:
        training_run_id = f"p2a-pusht-dinocular-s1-{producer}"
        training_card = by_id.get(training_run_id)
        reference = references.get(training_run_id)
        if training_card is None or reference is None:
            raise HarnessError(f"missing exact hashed P2a training card for {producer}")
        run_id = f"p2a-eval-pusht-dinocular-s1-{producer}"
        card = _base_card(
            spec,
            evidence,
            kind="p2a-open-loop",
            run_id=run_id,
            environment="pusht",
            arm="dinocular",
            seed=1,
            schema=LEGACY_RUN_CARD_SCHEMA,
        )
        card["fixed_manifest"] = copy.deepcopy(fixed_manifest)
        card["training_run_id"] = training_run_id
        card["training_run_dir"] = training_card["run_dir"]
        card["training_run_card"] = {
            "path": reference["path"],
            "file_sha256": reference["file_sha256"],
            "run_card_sha256": reference["run_card_sha256"],
        }
        card["segment_steps"] = training_card["segment_steps"]
        card["segment_sizing"] = copy.deepcopy(training_card["segment_sizing"])
        card["producer_pilot"] = copy.deepcopy(training_card["producer_pilot"])
        card["assumption_tags"] = copy.deepcopy(
            training_card.get("assumption_tags", [])
        )
        _depth_overrides(card, training_card["depth_inputs"])
        card["environment_variables"] = copy.deepcopy(
            training_card["environment_variables"]
        )
        card["depends_on"] = [training_run_id]
        finished = _finish_card(spec, card)
        if (
            finished["config_sha256"] != training_card["config_sha256"]
            or finished["overrides"] != training_card["overrides"]
        ):
            raise HarnessError("P2a evaluation configuration differs from training")
        cards.append(finished)
    return write_matrix(
        args.out,
        kind="p2a-open-loop",
        cards=cards,
        source_commit=evidence["source_commit"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "conf" / "study_matrix.yaml",
    )
    parser.add_argument("--contracts-index", type=Path)
    parser.add_argument("--launch-authorization", type=Path)
    parser.add_argument("--launch-authorization-sha256")
    parser.add_argument(
        "--local-code-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p2 = subparsers.add_parser("p2")
    p2.add_argument("--kind", choices=["geometry", "timing"], required=True)
    p2.add_argument("--producer", required=True)
    p2.add_argument("--out", type=Path, required=True)
    p2.set_defaults(function=make_p2)

    pilot = subparsers.add_parser("producer-pilot")
    pilot.add_argument("--env", required=True)
    pilot.add_argument("--seed", type=int, required=True)
    pilot.add_argument("--target-steps", type=int, required=True)
    pilot.add_argument("--encoder", default="dinocular")
    pilot.add_argument("--producers", required=True)
    pilot.add_argument("--rates", type=Path, required=True)
    pilot.add_argument("--out", type=Path, required=True)
    pilot.set_defaults(function=make_producer_pilot)

    training = subparsers.add_parser("training")
    training.add_argument("--encoders", required=True)
    training.add_argument("--envs", required=True)
    training.add_argument("--seeds", required=True)
    training.add_argument("--batch-size", type=int, required=True)
    training.add_argument("--target-steps", required=True)
    training.add_argument("--frameskips", required=True)
    training.add_argument("--predictor-lr", type=float, required=True)
    training.add_argument("--decoder", required=True)
    training.add_argument("--producer-decision", type=Path, required=True)
    training.add_argument("--rates", type=Path, required=True)
    training.add_argument("--heldout-manifests-dir", type=Path, required=True)
    training.add_argument("--out", type=Path, required=True)
    training.set_defaults(function=make_training)

    open_loop = subparsers.add_parser("open-loop")
    open_loop.add_argument("--training-matrix", type=Path, required=True)
    open_loop.add_argument("--manifests-dir", type=Path, required=True)
    open_loop.add_argument("--out", type=Path, required=True)
    open_loop.set_defaults(function=make_open_loop)

    pilot_eval = subparsers.add_parser("producer-pilot-eval")
    pilot_eval.add_argument("--training-matrix", type=Path, required=True)
    pilot_eval.add_argument("--manifests-dir", type=Path, required=True)
    pilot_eval.add_argument("--out", type=Path, required=True)
    pilot_eval.set_defaults(function=make_producer_pilot_eval)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "encoder", "dinocular") != "dinocular":
        raise HarnessError("P2a permits only the depth-on dinocular arm")
    matrix = args.function(args)
    print(json.dumps(matrix, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HarnessError as exc:
        print(f"HARNESS CONTRACT FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(2)
