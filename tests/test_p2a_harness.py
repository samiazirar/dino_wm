from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import lmdb
import numpy as np
import pytest
import torch
from torch import nn
import yaml
import zstandard

import training_timing
from datasets.depth_cache import DepthCacheError, DepthCacheReader
from depth_contract import (
    DepthContractError,
    canonical_json_bytes,
    load_native_depth_contract,
    sha256_bytes,
    sha256_file,
)
from models import dinocular_backbone
from models.dinocular import DinocularEncoder
from models.dinocular_backbone import BackendSpec
from eval_encoder_swap import (
    EvaluationContractError,
    _result_provenance,
    _validate_existing_result_provenance,
    _verify_training_completion,
)
from tools import make_manifests
from tools.harness_common import (
    HarnessError,
    LOCKED_ARMS,
    LOCKED_ENVS,
    LOCKED_HORIZONS,
    LOCKED_SEEDS,
    LOCKED_TARGETS,
    RUN_CARD_SCHEMA,
    derive_segment_sizing,
    finalize_run_card,
    load_matrix,
    validate_spec,
    verify_evaluation_bindings,
    write_matrix,
)
from tools.p2_harness_gate import _run_cem_horizon_five
from tools.run_matrix_card import _prepare_run_dir, evaluation_command
from tools.submit_matrix import (
    _validate_rates,
    verify_evaluation_training_dependencies,
)
from tools.collect_runs import CollectionError, _validate_open_loop_coverage


class TinyBackbone(nn.Module):
    def __init__(self, **_: object) -> None:
        super().__init__()
        self.rgb = nn.Conv2d(3, 8, 1)
        self.depth = nn.Conv2d(1, 8, 1)

    def forward_features(self, rgb, depth):
        value = (
            torch.nn.functional.adaptive_avg_pool2d(
                self.rgb(rgb) + self.depth(depth), (1, 1)
            )
            .flatten(2)
            .transpose(1, 2)
        )
        return {"x_norm_patchtokens": value, "x_norm_clstoken": value[:, 0]}


@pytest.fixture
def tiny_backend(monkeypatch):
    spec = BackendSpec(
        module=SimpleNamespace(Tiny=TinyBackbone),
        factories={"Tiny": 8},
        output_kind="dino_feature_dict",
        checkpoint_policy="exact",
    )
    monkeypatch.setitem(dinocular_backbone.BACKEND_REGISTRY, "tiny_native", spec)
    return spec


def _write_json(path: Path, value) -> str:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return sha256_file(path)


def _native_manifest(
    path: Path,
    *,
    checkpoint_sha256: str,
    producer_sha256: str,
    wire_sha256: str,
    wire_quantity: str = "raw_depth_z",
    wire_range=(0.0, 20.0),
    scale: float = 1.0,
    offset: float = 0.0,
    mean: float = 0.0,
    std: float = 1.0,
    neutral: float = -2.0,
) -> str:
    value = {
        "schema": "dinocular-native-depth-contract-v1",
        "status": "complete",
        "scientific_use_allowed": True,
        "checkpoint": {
            "sha256": checkpoint_sha256,
            "backend": "tiny_native",
            "factory": "Tiny",
            "checkpoint_key": "student",
            "state_prefix": "module.backbone.",
        },
        "producer": {
            "name": "future recovered native producer",
            "model": "native-depth-model",
            "version": "1",
            "code_commit": "a" * 40,
            "weight_sha256": "b" * 64,
            "temporal_mode": "complete_trajectory",
            "raw_units": "depth_z",
            "invocation": {"argv": ["producer", "--exact"]},
            "preprocessing": {"rgb": "uint8"},
            "scale": {"kind": "native"},
            "clipping": {"kind": "declared_per_binding"},
        },
        "encoder_input": {
            "checkpoint_native": {
                "quantity": "checkpoint_raw_depth_z",
                "units": "native_depth_z",
                "normalization": {
                    "kind": "affine_mean_std",
                    "mean": mean,
                    "std": std,
                },
            },
            "neutral": {"normalized_depth": neutral, "validity_mask": 0.0},
            "cache_bindings": [
                {
                    "producer_sha256": producer_sha256,
                    "wire_format_sha256": wire_sha256,
                    "wire_quantity": wire_quantity,
                    "wire_range": list(wire_range),
                    "affine_to_checkpoint_native": {
                        "operation": "checkpoint_native=wire*scale+offset",
                        "scale": scale,
                        "offset": offset,
                        "output_quantity": "checkpoint_raw_depth_z",
                    },
                    "clipping": {
                        "space": "checkpoint_native_before_normalization",
                        "minimum": -100.0,
                        "maximum": 100.0,
                    },
                    "interpolation": "bilinear_align_corners_false",
                    "invalid_mask": {
                        "source": "all_finite_cache_values",
                        "valid_value": 1.0,
                        "invalid_value": 0.0,
                    },
                }
            ],
        },
    }
    return _write_json(path, value)


def _tiny_encoder(
    tmp_path: Path,
    tiny_backend,
    *,
    scale: float,
    offset: float,
    wire_range,
    neutralize: bool = False,
):
    source = TinyBackbone()
    checkpoint = tmp_path / f"tiny-{scale}-{offset}-{neutralize}.pth"
    state = {
        f"module.backbone.{key}": value.clone()
        for key, value in source.state_dict().items()
    }
    torch.save({"student": state}, checkpoint)
    checkpoint_sha = sha256_file(checkpoint)
    producer_sha = "c" * 64
    wire_sha = "d" * 64
    native = tmp_path / f"native-{scale}-{offset}-{neutralize}.json"
    native_sha = _native_manifest(
        native,
        checkpoint_sha256=checkpoint_sha,
        producer_sha256=producer_sha,
        wire_sha256=wire_sha,
        wire_range=wire_range,
        scale=scale,
        offset=offset,
    )
    encoder = DinocularEncoder(
        backend="tiny_native",
        factory="Tiny",
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=checkpoint_sha,
        checkpoint_key="student",
        state_prefix="module.backbone.",
        input_size=32,
        num_patches=1,
        emb_dim=8,
        native_depth_contract_path=str(native),
        native_depth_contract_sha256=native_sha,
        selected_cache_producer_sha256=producer_sha,
        neutralize_depth_at_encoder_input=neutralize,
    )
    return encoder


def test_cache_wire_to_checkpoint_native_identity_and_calibrated_denormalization(
    tmp_path, tiny_backend
):
    identity = _tiny_encoder(
        tmp_path, tiny_backend, scale=1.0, offset=0.0, wire_range=(0.0, 20.0)
    )
    raw = torch.full((1, 1, 32, 32), 7.5)
    mask = torch.ones_like(raw)
    prepared, prepared_mask = identity.prepare_depth_encoder_input(raw, mask)
    torch.testing.assert_close(prepared, raw, rtol=0, atol=0)
    torch.testing.assert_close(prepared_mask, mask, rtol=0, atol=0)

    calibrated = _tiny_encoder(
        tmp_path, tiny_backend, scale=2.0, offset=10.0, wire_range=(0.0, 1.0)
    )
    normalized_wire = torch.full((1, 1, 32, 32), 0.5)
    prepared, _ = calibrated.prepare_depth_encoder_input(normalized_wire, mask)
    torch.testing.assert_close(
        prepared, torch.full_like(prepared, 11.0), rtol=0, atol=0
    )


def test_zero_boundary_and_missing_affine_fail_closed(tmp_path, tiny_backend):
    zero = _tiny_encoder(
        tmp_path,
        tiny_backend,
        scale=2.0,
        offset=10.0,
        wire_range=(0.0, 1.0),
        neutralize=True,
    )
    depth = torch.rand(2, 1, 32, 32)
    prepared, mask = zero.prepare_depth_encoder_input(depth, torch.ones_like(depth))
    torch.testing.assert_close(
        prepared, torch.full_like(prepared, -2.0), rtol=0, atol=0
    )
    torch.testing.assert_close(mask, torch.zeros_like(mask), rtol=0, atol=0)
    observed = []
    remove = zero.register_encoder_boundary_hook(
        lambda _encoder, boundary_depth, boundary_mask: observed.append(
            (boundary_depth.detach().clone(), boundary_mask.detach().clone())
        )
    )
    zero(
        torch.zeros(2, 3, 32, 32),
        depth,
        torch.ones_like(depth),
    )
    remove()
    assert len(observed) == 1
    torch.testing.assert_close(
        observed[0][0], torch.full_like(observed[0][0], -2.0), rtol=0, atol=0
    )
    torch.testing.assert_close(
        observed[0][1], torch.zeros_like(observed[0][1]), rtol=0, atol=0
    )

    manifest = json.loads(Path(zero.native_depth_contract_path).read_text())
    del manifest["encoder_input"]["cache_bindings"][0]["affine_to_checkpoint_native"]
    bad = tmp_path / "missing-affine.json"
    bad_sha = _write_json(bad, manifest)
    with pytest.raises(DepthContractError, match="affine"):
        load_native_depth_contract(
            bad, bad_sha, expected_checkpoint_sha256=zero.checkpoint_sha256
        )
    with pytest.raises(DepthContractError, match="no unique"):
        zero.native_depth_contract.binding_for("e" * 64)


def test_actual_cem_horizon_five_path_has_deformable_action_dim_four():
    class FakeWorldModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))

        def encode_obs(self, obs):
            visual = obs["visual"].mean(dim=(-1, -2, -3), keepdim=False).unsqueeze(-1)
            return {"visual": visual + self.anchor}

        def rollout(self, obs_0, act):
            history = self.encode_obs(obs_0)["visual"]
            increments = act.mean(dim=-1, keepdim=True).cumsum(dim=1)
            future = history[:, -1:] + increments
            result = {"visual": torch.cat([history, future], dim=1)}
            return result, result["visual"]

    model = FakeWorldModel()
    actions = _run_cem_horizon_five(
        model,
        {"visual": torch.zeros(1, 1, 3, 4, 4)},
        num_hist=1,
        action_dim=4,
    )
    assert tuple(actions.shape) == (1, 5, 4)


def _make_cache(tmp_path: Path):
    source_root = tmp_path / "raw"
    source = source_root / "wall_single" / "obses" / "episode_000.pth"
    source.parent.mkdir(parents=True)
    torch.save(torch.zeros(2, 3, 4, 4, dtype=torch.uint8), source)
    source_sha = sha256_file(source)
    cache = tmp_path / "wall.lmdb"
    cache.mkdir()
    producer = {"name": "synthetic", "version": "1", "settings": {"exact": True}}
    producer_sha = sha256_bytes(canonical_json_bytes(producer))
    wire = {
        "physical_key": "<split>/<episode:05d>/<frame:06d>",
        "dtype": "<f2",
        "shape": [224, 224],
        "order": "C",
        "compressor": "zstd",
        "compressor_level": 3,
        "map_size": 1 << 40,
        "normalization": "raw_depth_z",
        "inverted": False,
    }
    compressor = zstandard.ZstdCompressor(level=3)
    database = lmdb.open(str(cache), map_size=1 << 30)
    with database.begin(write=True) as transaction:
        for frame, scalar in enumerate((2.0, 3.0)):
            value = np.full((224, 224), scalar, dtype="<f2")
            transaction.put(
                f"valid/00000/{frame:06d}".encode(),
                compressor.compress(value.tobytes()),
            )
    database.sync(True)
    database.close()
    data_sha = sha256_file(cache / "data.mdb")
    source_index = [
        {
            "key": "valid/00000",
            "frames": 2,
            "source": "wall_single/obses/episode_000.pth",
            "sha256": source_sha,
        }
    ]
    cache_manifest = {
        "schema": "dinocular-depth-cache-v1",
        "manifest_id": "synthetic-cache",
        "environment": "wall",
        "trajectory_count": 1,
        "frame_count": 2,
        "source_index_sha256": sha256_bytes(canonical_json_bytes(source_index)),
        "producer": producer,
        "wire_format": wire,
        "closed_before_hash": True,
        "data_mdb_sha256": data_sha,
        "trajectories": [
            {
                "trajectory_key": "valid/00000",
                "source_path": "wall_single/obses/episode_000.pth",
                "source_video_sha256": source_sha,
                "ordered_frame_count": 2,
                "ordered_output_keys": [
                    "valid/00000/000000",
                    "valid/00000/000001",
                ],
            }
        ],
    }
    manifest_sha = _write_json(cache / "manifest.json", cache_manifest)
    validation = tmp_path / "validation.json"
    validation_sha = _write_json(
        validation,
        {
            "schema": "dinocular-depth-cache-validation-v1",
            "state": "PASS",
            "results": {
                "wall": {
                    "state": "PASS",
                    "manifest_id": "synthetic-cache",
                    "data_mdb_sha256": data_sha,
                }
            },
        },
    )
    native = tmp_path / "native-cache.json"
    native_sha = _native_manifest(
        native,
        checkpoint_sha256="e" * 64,
        producer_sha256=producer_sha,
        wire_sha256=sha256_bytes(canonical_json_bytes(wire)),
        wire_range=(0.0, 10.0),
    )
    return {
        "source_root": source_root,
        "cache": cache,
        "manifest_sha": manifest_sha,
        "validation": validation,
        "validation_sha": validation_sha,
        "native": native,
        "native_sha": native_sha,
        "producer_sha": producer_sha,
    }


def test_cache_alignment_physical_keys_shapes_and_hashes(tmp_path):
    fixture = _make_cache(tmp_path)
    reader = DepthCacheReader(
        environment="wall",
        source_root=fixture["source_root"],
        cache_dir=fixture["cache"],
        cache_manifest_sha256=fixture["manifest_sha"],
        validation_path=fixture["validation"],
        validation_sha256=fixture["validation_sha"],
        native_contract_path=fixture["native"],
        native_contract_sha256=fixture["native_sha"],
        expected_producer_sha256=fixture["producer_sha"],
        expected_checkpoint_sha256="e" * 64,
    )
    reader.assert_dataset_coverage([(None, 0, 2)])
    depth, validity = reader.read(split=None, episode=0, frames=[1, 0])
    assert depth.shape == (2, 224, 224)
    assert depth.dtype == torch.float32
    assert float(depth[0, 0, 0]) == 3.0
    assert float(depth[1, 0, 0]) == 2.0
    assert torch.all(validity == 1)
    with pytest.raises(DepthCacheError, match="missing cache trajectory"):
        reader.read(split="train", episode=0, frames=[0])


def test_real_student_configs_are_identical_except_neutral_switch():
    root = Path(__file__).resolve().parents[1]
    informative = yaml.safe_load((root / "conf/encoder/dinocular.yaml").read_text())
    neutral = yaml.safe_load(
        (root / "conf/encoder/dinocular_zerodepth.yaml").read_text()
    )
    assert informative.pop("neutralize_depth_at_encoder_input") is False
    assert neutral.pop("neutralize_depth_at_encoder_input") is True
    assert informative == neutral
    assert informative["checkpoint_sha256"] == (
        "decc7c73283bf46f66dbbedec6fb065ad0a943fb2ddb1c49317fafb0319f5dcc"
    )


def test_real_study_spec_accepts_exact_marvin_root_and_corrected_student_path():
    root = Path(__file__).resolve().parents[1]
    spec = yaml.safe_load((root / "conf/study_matrix.yaml").read_text())
    validate_spec(spec)
    assert spec["artifacts"]["dinocular_student"]["path"] == (
        "/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/"
        "checkpoints/dinov2_depthembed_dropout_fullpr.pth"
    )


def _timing_summary(path: Path, rate_overrides=None):
    rates = {
        f"{arm}/{environment}": {
            "state": "PASS",
            "optimizer_steps_per_second": 10.0,
        }
        for arm in LOCKED_ARMS
        for environment in LOCKED_ENVS
    }
    for key, rate in (rate_overrides or {}).items():
        rates[key]["optimizer_steps_per_second"] = float(rate)
    _write_json(
        path,
        {
            "schema": "dino-wm-p2-timing-summary-v1",
            "state": "PASS",
            "source_commit": "f" * 40,
            "matrix_sha256": "6" * 64,
            "rates": rates,
        },
    )
    return path


def _sizing(arm, environment, target_steps, rate=10.0):
    quantum = 1000
    derived = min(
        target_steps,
        int((rate * 8.0 * 3600.0 * 0.8) // quantum) * quantum,
    )
    return {
        "timing_summary_path": "/tmp/timing_summary.json",
        "timing_summary_sha256": "7" * 64,
        "timing_matrix_sha256": "6" * 64,
        "timing_source_commit": "f" * 40,
        "rate_key": f"{arm}/{environment}",
        "optimizer_steps_per_second": float(rate),
        "max_productive_hours": 8.0,
        "safety_margin_fraction": 0.2,
        "quantum_steps": quantum,
        "derived_segment_steps": derived,
    }


def _card(run_id, arm, environment, seed, kind="p3-training", gate_mode=None):
    sizing = _sizing(arm, environment, LOCKED_TARGETS[environment])
    card = {
        "schema": RUN_CARD_SCHEMA,
        "kind": kind,
        "run_id": run_id,
        "code_root": "/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/code/dino_wm",
        "run_dir": f"/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/outputs/{kind}/{run_id}",
        "source_commit": "f" * 40,
        "source_file_sha256": {"train.py": "a" * 64},
        "artifacts": {},
        "container": {
            "path": "/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/containers/test.sif",
            "sha256": "b" * 64,
        },
        "environment": environment,
        "arm": arm,
        "seed": seed,
        "target_steps": LOCKED_TARGETS[environment],
        "segment_steps": sizing["derived_segment_steps"],
        "segment_sizing": sizing,
        "frameskip": 5 if environment in {"pusht", "wall"} else 1,
        "horizons": LOCKED_HORIZONS[environment],
        "batch_size": 32,
        "predictor_lr": 0.00005,
        "decoder": False,
        "strict_resume": True,
        "depends_on": [],
        "environment_variables": {},
        "overrides": [f"env={environment}", f"encoder={arm}"],
        "config_sha256": "c" * 64,
    }
    if gate_mode is not None:
        card["gate_mode"] = gate_mode
    return finalize_run_card(card)


def test_locked_36_cell_matrix_hashing_and_dry_run(tmp_path):
    cards = [
        _card(f"p3-{environment}-{arm}-s{seed}", arm, environment, seed)
        for arm in LOCKED_ARMS
        for environment in LOCKED_ENVS
        for seed in LOCKED_SEEDS
    ]
    matrix_path = tmp_path / "matrix.yaml"
    write_matrix(matrix_path, kind="p3-training", cards=cards, source_commit="f" * 40)
    command = [
        "python3",
        str(Path(__file__).resolve().parents[1] / "tools/submit_matrix.py"),
        "train",
        "--matrix",
        str(matrix_path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    result = json.loads(completed.stdout)
    assert result["state"] == "PASS"
    assert result["job_count"] == 36
    assert result["sbatch_calls"] == 0
    assert all("submit_p3_chain.py" in job["command"][1] for job in result["jobs"])
    assert all("--segment-steps" in job["command"] for job in result["jobs"])


def test_p4_cards_bind_every_exact_hashed_p3_card_and_run_dir(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = yaml.safe_load((root / "conf/study_matrix.yaml").read_text())
    evidence = {
        "source_commit": "f" * 40,
        "source_file_sha256": {"train.py": "a" * 64},
        "artifacts": spec["artifacts"],
        "container": {
            "path": spec["container"]["path"],
            "sha256": spec["container"]["sha256"],
        },
    }
    p3_cards = []
    for arm in LOCKED_ARMS:
        for environment in LOCKED_ENVS:
            for seed in LOCKED_SEEDS:
                run_id = f"p3-{environment}-{arm}-s{seed}"
                card = make_manifests._base_card(
                    spec,
                    evidence,
                    kind="p3-training",
                    run_id=run_id,
                    environment=environment,
                    arm=arm,
                    seed=seed,
                )
                if arm != "dino_pinned":
                    make_manifests._depth_overrides(
                        card,
                        {
                            "producer": "approved-producer",
                            "producer_sha256": "1" * 64,
                            "cache_dir": f"{spec['study_root']}/depth/{environment}",
                            "cache_manifest_sha256": "2" * 64,
                            "validation_path": f"{spec['study_root']}/depth/{environment}.validation.json",
                            "validation_sha256": "3" * 64,
                            "native_contract_path": f"{spec['study_root']}/manifests/native.json",
                            "native_contract_sha256": "4" * 64,
                            "checkpoint_sha256": "5" * 64,
                        },
                    )
                card["segment_sizing"] = _sizing(arm, environment, card["target_steps"])
                card["segment_steps"] = card["segment_sizing"]["derived_segment_steps"]
                p3_cards.append(make_manifests._finish_card(spec, card))
    p3_matrix_path = tmp_path / "p3.yaml"
    p3_matrix = write_matrix(
        p3_matrix_path,
        kind="p3-training",
        cards=p3_cards,
        source_commit=evidence["source_commit"],
    )
    p3_by_id = {card["run_id"]: card for card in p3_cards}
    p3_refs = {reference["run_id"]: reference for reference in p3_matrix["cards"]}

    manifests = tmp_path / "fixed"
    manifests.mkdir()
    for environment in LOCKED_ENVS:
        path = manifests / f"openloop_{environment}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        _write_json(
            path.with_suffix(".meta.json"),
            {
                "schema": "dino-wm-open-loop-manifest-v1",
                "environment": environment,
                "frameskip": spec["environments"][environment]["frameskip"],
                "horizons": spec["environments"][environment]["horizons"],
                "manifest_sha256": sha256_file(path),
            },
        )

    monkeypatch.setattr(
        make_manifests,
        "_resolve_inputs",
        lambda _args: (spec, {}, evidence),
    )
    monkeypatch.setattr(
        make_manifests,
        "require_real_marvin_path",
        lambda value, _label: str(value),
    )
    p4_matrix_path = tmp_path / "p4.yaml"
    make_manifests.make_open_loop(
        SimpleNamespace(
            training_matrix=p3_matrix_path,
            manifests_dir=manifests,
            out=p4_matrix_path,
        )
    )
    _p4_matrix, p4_cards = load_matrix(p4_matrix_path)
    assert len(p4_cards) == 36
    for p4_card in p4_cards:
        p3_card = p3_by_id[p4_card["training_run_id"]]
        p3_reference = p3_refs[p4_card["training_run_id"]]
        assert p4_card["training_run_dir"] == p3_card["run_dir"]
        assert p4_card["training_run_card"] == {
            "path": p3_reference["path"],
            "file_sha256": p3_reference["file_sha256"],
            "run_card_sha256": p3_reference["run_card_sha256"],
        }
        assert p4_card.get("depth_inputs") == p3_card.get("depth_inputs")
        assert p4_card["environment_variables"] == p3_card["environment_variables"]
        expected_environment = {"DINOV2_REPO", "DINOV2_VITS14_WEIGHTS"}
        if p4_card["arm"] != "dino_pinned":
            expected_environment.update(
                {
                    "DINOCULAR_STUDENT_WEIGHTS",
                    "DINOCULAR_NATIVE_DEPTH_CONTRACT",
                    "DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256",
                    "DINOCULAR_CACHE_PRODUCER_SHA256",
                }
            )
        assert set(p4_card["environment_variables"]) == expected_environment
        assert p4_card["overrides"] == p3_card["overrides"]
        assert p4_card["config_sha256"] == p3_card["config_sha256"]

    p2a_cards = []
    for index, producer in enumerate(spec["p2a"]["producers"], 1):
        run_id = f"p2a-pusht-dinocular-s1-{producer}"
        card = make_manifests._base_card(
            spec,
            evidence,
            kind="p2a-producer-pilot",
            run_id=run_id,
            environment="pusht",
            arm="dinocular",
            seed=1,
        )
        make_manifests._depth_overrides(
            card,
            {
                "producer": producer,
                "producer_sha256": str(index) * 64,
                "cache_dir": f"{spec['study_root']}/depth/{producer}/pusht",
                "cache_manifest_sha256": "2" * 64,
                "validation_path": f"{spec['study_root']}/depth/{producer}/pusht.validation.json",
                "validation_sha256": "3" * 64,
                "native_contract_path": f"{spec['study_root']}/manifests/native.json",
                "native_contract_sha256": "4" * 64,
                "checkpoint_sha256": "5" * 64,
            },
        )
        card["segment_sizing"] = _sizing("dinocular", "pusht", card["target_steps"])
        card["segment_steps"] = card["segment_sizing"]["derived_segment_steps"]
        card["producer_pilot"] = {
            "producer": producer,
            "decision_horizons": [5, 10],
            "paired_manifest_required": True,
        }
        p2a_cards.append(make_manifests._finish_card(spec, card))
    p2a_training_path = tmp_path / "p2a.yaml"
    write_matrix(
        p2a_training_path,
        kind="p2a-producer-pilot",
        cards=p2a_cards,
        source_commit=evidence["source_commit"],
    )
    p2a_eval_path = tmp_path / "p2a_eval.yaml"
    make_manifests.make_producer_pilot_eval(
        SimpleNamespace(
            training_matrix=p2a_training_path,
            manifests_dir=manifests,
            out=p2a_eval_path,
        )
    )
    evaluation_matrix, evaluation_cards = load_matrix(p2a_eval_path)
    evaluation_refs = {item["run_id"]: item for item in evaluation_matrix["cards"]}
    assert len(evaluation_cards) == 2
    assert len({card["fixed_manifest"]["sha256"] for card in evaluation_cards}) == 1
    for evaluation_card in evaluation_cards:
        training_card, _metadata = verify_evaluation_bindings(evaluation_card)
        assert evaluation_card["depends_on"] == [training_card["run_id"]]
        command = evaluation_command(
            evaluation_card,
            Path(evaluation_refs[evaluation_card["run_id"]]["path"]),
        )
        assert command[1].endswith("eval_encoder_swap.py")
        assert command[-1].endswith("episode_errors.jsonl")
    dry_run = subprocess.run(
        [
            "python3",
            str(root / "tools/submit_matrix.py"),
            "producer-pilot-eval",
            "--matrix",
            str(p2a_eval_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    dry_result = json.loads(dry_run.stdout)
    assert dry_result["job_count"] == 2
    assert dry_result["sbatch_calls"] == 0
    assert all(
        "--dependency=afterok:<p2a-pusht-dinocular-s1-" in " ".join(job["command"])
        for job in dry_result["jobs"]
    )
    bad_config = dict(evaluation_cards[0])
    bad_config["config_sha256"] = "0" * 64
    with pytest.raises(Exception, match="exactly bound"):
        verify_evaluation_bindings(bad_config)
    bad_metadata = dict(evaluation_cards[0])
    bad_metadata["fixed_manifest"] = dict(bad_metadata["fixed_manifest"])
    bad_metadata["fixed_manifest"]["metadata_sha256"] = "0" * 64
    with pytest.raises(Exception, match="metadata hash"):
        verify_evaluation_bindings(bad_metadata)
    marker_card = dict(evaluation_cards[0])
    marker_card["run_dir"] = str(tmp_path / "retryable-evaluation")
    first_dir = _prepare_run_dir(marker_card)
    assert _prepare_run_dir(marker_card) == first_dir
    changed_marker = dict(marker_card)
    changed_marker["run_card_sha256"] = "0" * 64
    with pytest.raises(Exception, match="matching immutable marker"):
        _prepare_run_dir(changed_marker)
    training_card, _metadata = verify_evaluation_bindings(evaluation_cards[0])
    completion_dir = tmp_path / "completion"
    completion_dir.mkdir()
    checkpoint = completion_dir / "checkpoint.pth"
    torch.save(
        {"immutable_run_card_sha256": training_card["run_card_sha256"]},
        checkpoint,
    )
    progress = {
        "status": "TARGET_REACHED",
        "global_step": evaluation_cards[0]["target_steps"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_commit": evaluation_cards[0]["source_commit"],
        "immutable_run_card_sha256": training_card["run_card_sha256"],
    }
    _write_json(completion_dir / "progress.json", progress)
    _verify_training_completion(evaluation_cards[0], training_card, completion_dir)
    progress["immutable_run_card_sha256"] = "0" * 64
    _write_json(completion_dir / "progress.json", progress)
    with pytest.raises(EvaluationContractError, match="different run card"):
        _verify_training_completion(evaluation_cards[0], training_card, completion_dir)


def test_evaluation_execute_requires_passed_chain_tail_dependency(tmp_path):
    training_run_id = "p3-pusht-dinocular-s1"
    run_card_sha256 = "a" * 64
    source_commit = "f" * 40
    target_steps = 7
    training_run_dir = tmp_path / "training"
    training_run_card = tmp_path / "training_run_card.yaml"
    training_run_card.write_text("immutable: true\n", encoding="utf-8")
    checkpoint_dir = training_run_dir / "checkpoints" / "steps"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / f"step_{target_steps:09d}.pth"
    torch.save(
        {
            "global_step": target_steps,
            "immutable_run_card_sha256": run_card_sha256,
        },
        checkpoint,
    )
    progress = {
        "status": "TARGET_REACHED",
        "global_step": target_steps,
        "target_steps": target_steps,
        "source_commit": source_commit,
        "immutable_run_card_sha256": run_card_sha256,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "sampler": {"next_step": target_steps},
    }
    _write_json(training_run_dir / "progress.json", progress)
    chain = {
        "schema": "dino-wm.p3-slurm-chain.v1",
        "status": "PASSED",
        "run_dir": str(training_run_dir),
        "target_steps": target_steps,
        "run_card_sha256": run_card_sha256,
        "run_card": str(training_run_card),
        "run_card_file_sha256": sha256_file(training_run_card),
        "source_commit": source_commit,
        "jobs": [{"job_id": "111"}, {"job_id": "222"}],
        "events": [
            {
                "job_id": "222",
                "progress_status": "TARGET_REACHED",
                "global_step": target_steps,
                "immutable_run_card_sha256": run_card_sha256,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        ],
        "final_progress": progress,
    }
    chain_path = training_run_dir / "chain.json"
    _write_json(chain_path, chain)
    evaluation_card = {
        "run_id": "p4-pusht-dinocular-s1",
        "training_run_id": training_run_id,
        "training_run_dir": str(training_run_dir),
        "training_run_card": {
            "path": str(training_run_card),
            "file_sha256": sha256_file(training_run_card),
            "run_card_sha256": run_card_sha256,
        },
        "depends_on": [training_run_id],
        "target_steps": target_steps,
        "source_commit": source_commit,
    }

    with pytest.raises(HarnessError, match="is not chain tail 222"):
        verify_evaluation_training_dependencies(
            [evaluation_card], {training_run_id: "111"}
        )
    verify_evaluation_training_dependencies([evaluation_card], {training_run_id: "222"})

    chain["status"] = "CHAINED"
    _write_json(chain_path, chain)
    with pytest.raises(HarnessError, match="not exact and PASSED"):
        verify_evaluation_training_dependencies(
            [evaluation_card], {training_run_id: "222"}
        )
    chain["status"] = "PASSED"
    _write_json(chain_path, chain)
    changed_progress = dict(progress)
    changed_progress["checkpoint_sha256"] = "0" * 64
    _write_json(training_run_dir / "progress.json", changed_progress)
    with pytest.raises(HarnessError, match="not exact TARGET_REACHED"):
        verify_evaluation_training_dependencies(
            [evaluation_card], {training_run_id: "222"}
        )


def test_p2_cli_locks_and_actual_gate_modes(tmp_path):
    cards = []
    for arm in LOCKED_ARMS:
        for environment in LOCKED_ENVS:
            card = _card(
                f"p2-geometry-{arm}-{environment}-s1",
                arm,
                environment,
                1,
                kind="p2-geometry",
                gate_mode="geometry",
            )
            card = dict(card)
            card["target_steps"] = 1
            card["segment_steps"] = 1
            card = finalize_run_card(card)
            cards.append(card)
    matrix_path = tmp_path / "p2.yaml"
    write_matrix(matrix_path, kind="p2-geometry", cards=cards, source_commit="f" * 40)
    tool = Path(__file__).resolve().parents[1] / "tools/submit_matrix.py"
    base = [
        "python3",
        str(tool),
        "canary",
        "--matrix",
        str(matrix_path),
        "--encoders",
        "dino_pinned,dinocular,dinocular_zerodepth",
        "--envs",
        "pusht,wall,rope,granular",
        "--seeds",
        "1",
        "--minutes",
        "30",
        "--frameskips",
        "pusht=5,wall=5,rope=1,granular=1",
    ]
    result = json.loads(
        subprocess.run(base, check=True, capture_output=True, text=True).stdout
    )
    assert result["job_count"] == 12
    failed = subprocess.run(
        base[:-1] + ["pusht=5,wall=5,rope=5,granular=5"], capture_output=True, text=True
    )
    assert failed.returncode == 2


def test_all_run_card_container_invocations_forward_dinocular_environment():
    root = Path(__file__).resolve().parents[1]
    p3_wrapper = (root / "tools/p3_step_segment.sbatch").read_text()
    matrix_wrapper = (root / "tools/matrix_job.sbatch").read_text()
    helper_path = "$PROJECT/code/dino_wm/tools/dinocular_container_env.sh"
    assert f'source "{helper_path}"' in p3_wrapper
    assert f'source "{helper_path}"' in matrix_wrapper
    assert p3_wrapper.count('"${DINOCULAR_CONTAINER_ENV[@]}"') == 3
    geometry_block = p3_wrapper.split("p2-geometry)", 1)[1].split(";;", 1)[0]
    timing_block = p3_wrapper.split("p2-timing)", 1)[1].split(";;", 1)[0]
    assert '"${DINOCULAR_CONTAINER_ENV[@]}"' in geometry_block
    assert '"${DINOCULAR_CONTAINER_ENV[@]}"' in timing_block
    assert matrix_wrapper.count('"${DINOCULAR_CONTAINER_ENV[@]}"') == 1
    assert "--field arm" in matrix_wrapper


DINOCULAR_TEST_ENVIRONMENT = {
    "DINOCULAR_STUDENT_WEIGHTS": "/pinned/student.pth",
    "DINOCULAR_NATIVE_DEPTH_CONTRACT": "/pinned/native.json",
    "DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256": "a" * 64,
    "DINOCULAR_CACHE_PRODUCER_SHA256": "b" * 64,
}


def _run_container_environment_helper(arm, environment, *, check):
    helper = Path(__file__).resolve().parents[1] / "tools/dinocular_container_env.sh"
    return subprocess.run(
        [
            "bash",
            "-c",
            (
                'source "$1"; build_dinocular_container_env "$2"; '
                "if ((${#DINOCULAR_CONTAINER_ENV[@]})); then "
                'printf "%s\\n" "${DINOCULAR_CONTAINER_ENV[@]}"; fi'
            ),
            "bash",
            str(helper),
            arm,
        ],
        env=environment,
        check=check,
        capture_output=True,
        text=True,
    )


def test_dino_pinned_container_environment_needs_no_dinocular_variables():
    environment = os.environ.copy()
    for key in DINOCULAR_TEST_ENVIRONMENT:
        environment.pop(key, None)
    completed = _run_container_environment_helper(
        "dino_pinned", environment, check=True
    )
    assert completed.stdout == ""


@pytest.mark.parametrize("arm", ["dinocular", "dinocular_zerodepth"])
def test_dinocular_arms_container_environment_contains_exact_values(arm):
    environment = os.environ.copy()
    environment.update(DINOCULAR_TEST_ENVIRONMENT)
    completed = _run_container_environment_helper(arm, environment, check=True)
    assert completed.stdout.splitlines() == [
        value
        for key, path in DINOCULAR_TEST_ENVIRONMENT.items()
        for value in ("--env", f"{key}={path}")
    ]


@pytest.mark.parametrize("arm", ["dinocular", "dinocular_zerodepth"])
@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("DINOCULAR_STUDENT_WEIGHTS", "student weights are required"),
        ("DINOCULAR_NATIVE_DEPTH_CONTRACT", "native contract is required"),
        (
            "DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256",
            "native contract hash is required",
        ),
        ("DINOCULAR_CACHE_PRODUCER_SHA256", "producer hash is required"),
    ],
)
def test_dinocular_arms_container_environment_fails_closed_when_missing(
    arm, missing, message
):
    environment = os.environ.copy()
    environment.update(DINOCULAR_TEST_ENVIRONMENT)
    environment.pop(missing)
    completed = _run_container_environment_helper(arm, environment, check=False)
    assert completed.returncode != 0
    assert message in completed.stderr


TIMING_WINDOWS = {
    "pusht": 1_981_721,
    "wall": 70_848,
    "rope": 17_100,
    "granular": 17_100,
}
TIMING_PRODUCER_SHA256 = "d" * 64
TIMING_NATIVE_SHA256 = "e" * 64


def test_timing_runtime_binds_immutable_card_and_dinocular_identities(
    tmp_path, monkeypatch
):
    class ReleasedDataset:
        @staticmethod
        def __len__():
            return TIMING_WINDOWS["pusht"]

    code_root = tmp_path / "code"
    code_root.mkdir()
    train_path = code_root / "train.py"
    resume_path = code_root / "training_resume.py"
    wrapper_path = code_root / "p3_step_segment.sbatch"
    for path, text in (
        (train_path, "train\n"),
        (resume_path, "resume\n"),
        (wrapper_path, "wrapper\n"),
    ):
        path.write_text(text, encoding="utf-8")
    dinov2_path = tmp_path / "dinov2.pth"
    student_path = tmp_path / "student.pth"
    native_path = tmp_path / "native.json"
    dinov2_path.write_bytes(b"dino")
    student_path.write_bytes(b"student")
    native_path.write_text("{}\n", encoding="utf-8")
    output_path = tmp_path / "timing_result.json"
    runtime_path = tmp_path / "timing_runtime_card.yaml"
    immutable_path = tmp_path / "immutable.yaml"
    source_hashes = {
        "train.py": sha256_file(train_path),
        "training_resume.py": sha256_file(resume_path),
        "training_timing.py": sha256_file(Path(training_timing.__file__)),
        "tools/p3_step_segment.sbatch": sha256_file(wrapper_path),
    }
    producer_sha256 = "a" * 64
    immutable = finalize_run_card(
        {
            "schema": RUN_CARD_SCHEMA,
            "kind": "p2-timing",
            "gate_mode": "timing",
            "run_id": "p2-timing-dinocular-pusht-s1",
            "run_dir": str(tmp_path),
            "source_commit": "f" * 40,
            "source_file_sha256": source_hashes,
            "artifacts": {
                "dinov2": {
                    "path": str(dinov2_path),
                    "sha256": sha256_file(dinov2_path),
                },
                "dinocular_student": {
                    "path": str(student_path),
                    "sha256": sha256_file(student_path),
                },
            },
            "container": {"path": "/tmp/test.sif", "sha256": "b" * 64},
            "depth_inputs": {
                "native_contract_path": str(native_path),
                "native_contract_sha256": sha256_file(native_path),
                "producer_sha256": producer_sha256,
                "checkpoint_sha256": sha256_file(student_path),
            },
            "environment": "pusht",
            "arm": "dinocular",
            "seed": 1,
            "target_steps": 220,
            "segment_steps": 220,
            "frameskip": 5,
            "batch_size": 32,
            "decoder": False,
            "timing": {"fixed_steps": 200, "warmup_steps": 20},
        }
    )
    immutable_path.write_text(
        yaml.safe_dump(immutable, sort_keys=False), encoding="utf-8"
    )
    for key, value in {
        "STRICT_P2_ENV": "pusht",
        "STRICT_P2_ARM": "dinocular",
        "STRICT_P2_CONTAINER": "/tmp/test.sif",
        "STRICT_P2_CONTAINER_SHA256": "b" * 64,
        "STRICT_P2_SLURM_WRAPPER": str(wrapper_path),
        "STRICT_P2_IMMUTABLE_RUN_CARD": str(immutable_path),
        "STRICT_P2_IMMUTABLE_RUN_CARD_SHA256": immutable["run_card_sha256"],
        "DINOV2_VITS14_WEIGHTS": str(dinov2_path),
        "DINOCULAR_STUDENT_WEIGHTS": str(student_path),
        "DINOCULAR_NATIVE_DEPTH_CONTRACT": str(native_path),
        "DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256": sha256_file(native_path),
        "DINOCULAR_CACHE_PRODUCER_SHA256": producer_sha256,
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(training_timing.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(training_timing.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        training_timing.torch.cuda,
        "get_device_name",
        lambda _device: "NVIDIA A100-SXM4-80GB",
    )
    monkeypatch.setattr(training_timing.NvidiaSmiMonitor, "start", lambda _self: None)
    cfg = training_timing.OmegaConf.create(
        {
            "training": {
                "timing_output": str(output_path),
                "timing_run_card": str(runtime_path),
                "timing_warmup_steps": 20,
                "timing_measured_steps": 200,
                "timing_projection_target_steps": LOCKED_TARGETS["pusht"],
                "strict_determinism": True,
                "target_steps": 220,
                "batch_size": 32,
                "resume_from": None,
                "test_signal_after_step": None,
            },
            "env": {"name": "pusht", "num_workers": 0},
            "encoder": {"name": "dinocular"},
            "frameskip": 5,
        }
    )
    trainer = SimpleNamespace(
        cfg=cfg,
        accelerator=SimpleNamespace(num_processes=1),
        global_step=0,
        device="cuda:0",
        datasets={"train": ReleasedDataset()},
        base_path=code_root,
        source_commit="f" * 40,
        dataset_order_sha256="c" * 64,
        resume_config_sha256="d" * 64,
    )
    sampler = SimpleNamespace(steps_per_epoch=(TIMING_WINDOWS["pusht"] + 31) // 32)
    window = training_timing.StrictTimingWindow(
        trainer,
        sampler=sampler,
        segment_start=0,
        segment_stop=220,
        checkpoint_every=0,
    )
    runtime = yaml.safe_load(runtime_path.read_text())
    assert runtime["claim"] == "strict production-path dinocular single-A100 timing"
    assert runtime["immutable_run_card"] == window.immutable_run_card
    assert runtime["artifacts"]["dinocular_identity"] == {
        "student": immutable["artifacts"]["dinocular_student"],
        "native_depth_contract": {
            "path": str(native_path),
            "sha256": sha256_file(native_path),
        },
        "selected_producer_sha256": producer_sha256,
    }
    monkeypatch.setenv("DINOCULAR_CACHE_PRODUCER_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="student, native contract, or producer"):
        training_timing.StrictTimingWindow(
            trainer,
            sampler=sampler,
            segment_start=0,
            segment_stop=220,
            checkpoint_every=0,
        )


def _timing_collector_fixture(tmp_path):
    cards = []
    results_root = tmp_path / "timing-results"
    artifact_root = "/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm"
    for arm in LOCKED_ARMS:
        for environment in LOCKED_ENVS:
            run_id = f"p2-timing-{arm}-{environment}-s1"
            card = _card(
                run_id,
                arm,
                environment,
                1,
                kind="p2-timing",
                gate_mode="timing",
            )
            card = dict(card)
            card["target_steps"] = 220
            card["segment_steps"] = 220
            card["timing"] = {"fixed_steps": 200, "warmup_steps": 20}
            card["artifacts"] = {
                "dinov2": {
                    "path": f"{artifact_root}/models/dinov2_vits14_pretrain.pth",
                    "sha256": "b" * 64,
                },
                "dinocular_student": {
                    "path": f"{artifact_root}/checkpoints/student.pth",
                    "sha256": "c" * 64,
                },
            }
            if arm != "dino_pinned":
                card["depth_inputs"] = {
                    "native_contract_path": f"{artifact_root}/manifests/native.json",
                    "native_contract_sha256": TIMING_NATIVE_SHA256,
                    "producer_sha256": TIMING_PRODUCER_SHA256,
                    "checkpoint_sha256": "c" * 64,
                }
            card = finalize_run_card(card)
            cards.append(card)
    matrix_path = tmp_path / "timing.yaml"
    matrix = write_matrix(
        matrix_path,
        kind="p2-timing",
        cards=cards,
        source_commit="f" * 40,
    )
    references = {item["run_id"]: item for item in matrix["cards"]}
    for card in cards:
        arm = card["arm"]
        environment = card["environment"]
        run_id = card["run_id"]
        run_dir = results_root / run_id
        run_dir.mkdir(parents=True)
        result_path = run_dir / "timing_result.json"
        reference = references[run_id]
        immutable = {
            "path": str(Path(reference["path"]).resolve()),
            "file_sha256": reference["file_sha256"],
            "run_card_sha256": reference["run_card_sha256"],
            "run_id": run_id,
        }
        dinocular_identity = None
        if arm != "dino_pinned":
            dinocular_identity = {
                "student": card["artifacts"]["dinocular_student"],
                "native_depth_contract": {
                    "path": card["depth_inputs"]["native_contract_path"],
                    "sha256": card["depth_inputs"]["native_contract_sha256"],
                },
                "selected_producer_sha256": card["depth_inputs"]["producer_sha256"],
            }
        artifacts = {
            "source_commit": card["source_commit"],
            "container_sha256": card["container"]["sha256"],
            "dinov2_weights_path": card["artifacts"]["dinov2"]["path"],
            "dinov2_weights_sha256": card["artifacts"]["dinov2"]["sha256"],
            "dinocular_identity": dinocular_identity,
        }
        windows = TIMING_WINDOWS[environment]
        steps_per_epoch = (windows + 31) // 32
        _write_json(
            result_path,
            {
                "schema": "dino-wm.strict-p2-timing.v1",
                "status": "MEASURED_PASS",
                "arm": arm,
                "environment": environment,
                "measurement_claim": f"strict production-path {arm} single-A100 timing",
                "projection_status": "PROJECTED_FROM_MEASURED_ARM_SPECIFIC_RATE",
                "assumption_tags": [],
                "immutable_run_card": immutable,
                "slurm_job_id": "12345",
                "gpu": "NVIDIA A100-SXM4-80GB",
                "gpu_count": 1,
                "artifacts": artifacts,
                "strict_settings": {
                    "deterministic_algorithms": True,
                    "cudnn_benchmark": False,
                    "cudnn_deterministic": True,
                    "cuda_matmul_allow_tf32": False,
                    "cudnn_allow_tf32": False,
                    "cublas_workspace_config": ":4096:8",
                    "processes": 1,
                    "num_workers": 0,
                },
                "config": {
                    "training": {
                        "seed": 1,
                        "predictor_lr": 0.00005,
                        "strict_determinism": True,
                        "target_steps": 220,
                        "segment_steps": 220,
                        "checkpoint_every_steps": 0,
                        "resume_from": None,
                        "batch_size": 32,
                        "timing_warmup_steps": 20,
                        "timing_measured_steps": 200,
                        "timing_projection_target_steps": LOCKED_TARGETS[environment],
                    },
                    "env": {"num_workers": 0},
                    "model": {
                        "train_encoder": False,
                        "train_predictor": True,
                        "train_decoder": False,
                    },
                    "frameskip": card["frameskip"],
                    "has_decoder": False,
                },
                "warmup_steps_excluded": 20,
                "measured_steps": 200,
                "measured_samples": 6400,
                "global_batch_size": 32,
                "frame_skip": card["frameskip"],
                "projection_target_steps": LOCKED_TARGETS[environment],
                "steps_per_second": 1.25,
                "samples_per_second": 40.0,
                "measured_seconds": 160.0,
                "final_loss": 1.0,
                "peak_torch_reserved_mib": 1024.0,
                "train_windows": windows,
                "steps_per_epoch": steps_per_epoch,
                "sampler_state_after_window": {
                    "dataset_size": windows,
                    "batch_size": 32,
                    "steps_per_epoch": steps_per_epoch,
                    "next_step": 220,
                },
            },
        )
        runtime_path = run_dir / "timing_runtime_card.yaml"
        runtime_path.write_text(
            yaml.safe_dump(
                {
                    "schema": "dino-wm.strict-p2-run-card.v1",
                    "status": "PASSED",
                    "arm": arm,
                    "environment": environment,
                    "claim": f"strict production-path {arm} single-A100 timing",
                    "measurement_status": "MEASURED_ARM_SPECIFIC_AFTER_COMPLETION",
                    "immutable_run_card": immutable,
                    "protocol": {
                        "processes": 1,
                        "global_batch_size": 32,
                        "num_workers": 0,
                        "frame_skip": card["frameskip"],
                        "warmup_steps_excluded": 20,
                        "measured_optimizer_steps": 200,
                        "projection_target_steps": LOCKED_TARGETS[environment],
                        "checkpointing_in_measured_window": False,
                        "evaluation_in_measured_window": False,
                        "profiler_in_measured_window": False,
                    },
                    "artifacts": artifacts,
                    "projection": {
                        "target_steps": LOCKED_TARGETS[environment],
                        "status": "PROJECTED_FROM_MEASURED_ARM_SPECIFIC_RATE",
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        _write_json(
            run_dir / "timing_gate.json",
            {
                "schema": "dino-wm-p2-timing-gate-v1",
                "state": "PASS",
                "run_id": run_id,
                "result_path": str(result_path),
                "result_sha256": sha256_file(result_path),
                "runtime_card_path": str(runtime_path),
                "runtime_card_sha256": sha256_file(runtime_path),
            },
        )
    summary_path = tmp_path / "timing_summary.json"
    return matrix_path, results_root, summary_path


def _run_timing_collector(matrix_path, results_root, summary_path, *, check=True):
    return subprocess.run(
        [
            "python3",
            str(Path(__file__).resolve().parents[1] / "tools/collect_p2_timing.py"),
            "--matrix",
            str(matrix_path),
            "--results-root",
            str(results_root),
            "--out",
            str(summary_path),
        ],
        check=check,
        capture_output=True,
        text=True,
    )


def _run_timing_gate(matrix_path, results_root, run_id, *, check=True):
    matrix = yaml.safe_load(matrix_path.read_text())
    run_card = next(
        item["path"] for item in matrix["cards"] if item["run_id"] == run_id
    )
    return subprocess.run(
        [
            "python3",
            str(Path(__file__).resolve().parents[1] / "tools/p2_harness_gate.py"),
            "timing",
            "--run-card",
            run_card,
            "--run-dir",
            str(results_root / run_id),
        ],
        check=check,
        capture_output=True,
        text=True,
    )


def _rewrite_timing_result(results_root, run_id, path, value):
    result_path = results_root / run_id / "timing_result.json"
    result = json.loads(result_path.read_text())
    target = result
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _write_json(result_path, result)
    gate_path = results_root / run_id / "timing_gate.json"
    gate = json.loads(gate_path.read_text())
    gate["result_sha256"] = sha256_file(result_path)
    _write_json(gate_path, gate)


def test_twelve_card_timing_collector_emits_submit_consumable_rates(tmp_path):
    matrix_path, results_root, summary_path = _timing_collector_fixture(tmp_path)
    for arm in LOCKED_ARMS:
        for environment in LOCKED_ENVS:
            _run_timing_gate(
                matrix_path,
                results_root,
                f"p2-timing-{arm}-{environment}-s1",
            )
    _run_timing_collector(matrix_path, results_root, summary_path)
    summary = json.loads(summary_path.read_text())
    assert summary["state"] == "PASS"
    assert set(summary["rates"]) == {
        f"{arm}/{environment}" for arm in LOCKED_ARMS for environment in LOCKED_ENVS
    }
    assert all(
        record["optimizer_steps_per_second"] == 1.25
        for record in summary["rates"].values()
    )
    training_card = _card("p3-pusht-dinocular-s1", "dinocular", "pusht", 1)
    training_card = dict(training_card)
    training_card["segment_sizing"] = derive_segment_sizing(
        summary_path=summary_path,
        arm="dinocular",
        environment="pusht",
        target_steps=123858,
        policy={
            "max_productive_hours": 8.0,
            "safety_margin_fraction": 0.2,
            "quantum_steps": 1000,
        },
    )
    training_card["segment_steps"] = training_card["segment_sizing"][
        "derived_segment_steps"
    ]
    _validate_rates(summary_path, [training_card], 8.0)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("train_windows",), 1_981_721.0, "must be integer 1981721"),
        (("measured_samples",), 6399, "must be integer 6400"),
        (("gpu_count",), 2, "must be integer 1"),
        (("gpu",), "NVIDIA H100 80GB HBM3", "one scheduled A100"),
        (
            ("strict_settings", "deterministic_algorithms"),
            False,
            "strict deterministic runtime settings differ",
        ),
        (
            ("strict_settings", "cuda_matmul_allow_tf32"),
            True,
            "strict deterministic runtime settings differ",
        ),
        (
            ("strict_settings", "cudnn_allow_tf32"),
            True,
            "strict deterministic runtime settings differ",
        ),
        (
            ("strict_settings", "cublas_workspace_config"),
            ":16:8",
            "strict deterministic runtime settings differ",
        ),
        (
            ("strict_settings", "processes"),
            2,
            "strict deterministic runtime settings differ",
        ),
        (
            ("strict_settings", "num_workers"),
            1,
            "strict deterministic runtime settings differ",
        ),
    ],
)
def test_timing_collector_rejects_mismatched_counts_and_runtime_settings(
    tmp_path, path, value, message
):
    matrix_path, results_root, summary_path = _timing_collector_fixture(tmp_path)
    _rewrite_timing_result(
        results_root,
        "p2-timing-dino_pinned-pusht-s1",
        path,
        value,
    )
    completed = _run_timing_collector(
        matrix_path, results_root, summary_path, check=False
    )
    assert completed.returncode == 2
    assert message in completed.stderr


def test_timing_collector_rejects_run_card_and_dinocular_identity_drift(tmp_path):
    matrix_path, results_root, summary_path = _timing_collector_fixture(tmp_path)
    run_id = "p2-timing-dinocular-pusht-s1"
    result_path = results_root / run_id / "timing_result.json"
    runtime_path = results_root / run_id / "timing_runtime_card.yaml"
    result = json.loads(result_path.read_text())
    runtime = yaml.safe_load(runtime_path.read_text())
    result["immutable_run_card"]["run_card_sha256"] = "0" * 64
    runtime["immutable_run_card"]["run_card_sha256"] = "0" * 64
    _write_json(result_path, result)
    runtime_path.write_text(yaml.safe_dump(runtime, sort_keys=False), encoding="utf-8")
    gate_path = results_root / run_id / "timing_gate.json"
    gate = json.loads(gate_path.read_text())
    gate["result_sha256"] = sha256_file(result_path)
    gate["runtime_card_sha256"] = sha256_file(runtime_path)
    _write_json(gate_path, gate)
    completed = _run_timing_collector(
        matrix_path, results_root, summary_path, check=False
    )
    assert completed.returncode == 2
    assert "not bound to the exact run card" in completed.stderr

    result["immutable_run_card"]["run_card_sha256"] = next(
        item["run_card_sha256"]
        for item in yaml.safe_load(matrix_path.read_text())["cards"]
        if item["run_id"] == run_id
    )
    runtime["immutable_run_card"] = result["immutable_run_card"]
    result["artifacts"]["dinocular_identity"]["selected_producer_sha256"] = "1" * 64
    runtime["artifacts"] = result["artifacts"]
    _write_json(result_path, result)
    runtime_path.write_text(yaml.safe_dump(runtime, sort_keys=False), encoding="utf-8")
    gate["result_sha256"] = sha256_file(result_path)
    gate["runtime_card_sha256"] = sha256_file(runtime_path)
    _write_json(gate_path, gate)
    completed = _run_timing_collector(
        matrix_path, results_root, summary_path, check=False
    )
    assert completed.returncode == 2
    assert "student/native-contract/producer identity differs" in completed.stderr


def test_timing_gate_rejects_truncating_float_train_windows(tmp_path):
    matrix_path, results_root, _summary_path = _timing_collector_fixture(tmp_path)
    run_id = "p2-timing-dino_pinned-pusht-s1"
    _rewrite_timing_result(results_root, run_id, ("train_windows",), 1_981_721.0)
    completed = _run_timing_gate(matrix_path, results_root, run_id, check=False)
    assert completed.returncode == 2
    assert "must be integer 1981721" in completed.stderr


def test_pusht_segment_budget_rejects_26000_and_derives_safe_chunk(tmp_path):
    card = _card("p3-pusht-dino_pinned-s1", "dino_pinned", "pusht", 1)
    rates = _timing_summary(
        tmp_path / "rates.json",
        {"dino_pinned/pusht": 0.488031},
    )
    derived = derive_segment_sizing(
        summary_path=rates,
        arm="dino_pinned",
        environment="pusht",
        target_steps=123858,
        policy={
            "max_productive_hours": 8.0,
            "safety_margin_fraction": 0.2,
            "quantum_steps": 1000,
        },
    )
    assert 26000 / 0.488031 / 3600 == pytest.approx(14.7986956202)
    assert derived["derived_segment_steps"] == 11000
    unsafe = dict(card)
    unsafe["segment_steps"] = 26000
    unsafe["segment_sizing"] = derived
    with pytest.raises(Exception, match="not bound"):
        _validate_rates(rates, [unsafe], 8.0)
    safe = dict(card)
    safe["segment_steps"] = derived["derived_segment_steps"]
    safe["segment_sizing"] = derived
    _validate_rates(rates, [safe], 8.0)


def _provenance_fields(
    *,
    arm: str,
    token: str,
    slurm_job_id: str = "12345",
    depth_token: str = "a",
):
    depth = arm != "dino_pinned"
    return {
        "evaluation_run_card_sha256": token * 64,
        "evaluation_run_card_file_sha256": token * 64,
        "training_run_card_sha256": token * 64,
        "training_run_card_file_sha256": token * 64,
        "source_commit": "f" * 40,
        "config_sha256": token * 64,
        "container_sha256": "c" * 64,
        "checkpoint_sha256": token * 64,
        "manifest_sha256": "9" * 64,
        "slurm_job_id": slurm_job_id,
        "depth_producer_sha256": depth_token * 64 if depth else None,
        "depth_cache_manifest_sha256": depth_token * 64 if depth else None,
        "depth_native_contract_sha256": depth_token * 64 if depth else None,
        "depth_validation_sha256": depth_token * 64 if depth else None,
        "depth_checkpoint_sha256": depth_token * 64 if depth else None,
    }


def _result_row(producer, episode, model5, persistence5, model10, persistence10):
    token = "a" if producer == "da3_giant_video" else "b"
    return {
        "schema": "dino-wm-open-loop-episode-errors-v1",
        "run_id": f"p2a-{producer}",
        "arm": "dinocular",
        "producer": producer,
        "seed": 1,
        "environment": "pusht",
        "episode": episode,
        **_provenance_fields(arm="dinocular", token=token, depth_token=token),
        "manifest_keys": [f"pusht/valid/{episode:05d}/000000"],
        "manifest_key_count": 1,
        "horizons": {
            "5": {
                "model_squared_error": model5,
                "persistence_squared_error": persistence5,
                "element_count": 10,
                "raw_mse": model5 / 10,
            },
            "10": {
                "model_squared_error": model10,
                "persistence_squared_error": persistence10,
                "element_count": 10,
                "raw_mse": model10 / 10,
            },
        },
        "horizon_auc_raw_mse": 0.0,
    }


def test_p4_collector_rejects_manifest_hash_key_or_horizon_mismatch():
    groups = {}
    for arm_index, arm in enumerate(LOCKED_ARMS):
        for seed_index, seed in enumerate(LOCKED_SEEDS):
            token = "123456789"[arm_index * len(LOCKED_SEEDS) + seed_index]
            row = {
                "run_id": f"p4-wall-{arm}-s{seed}",
                "environment": "wall",
                "arm": arm,
                "seed": seed,
                "episode": 0,
                **_provenance_fields(arm=arm, token=token),
                "manifest_keys": ["wall/valid/00000/000000"],
                "horizons": {str(value): {} for value in (1, 5, 10)},
            }
            groups[("wall", arm, seed)] = [row]
    _validate_open_loop_coverage(groups)
    for field, value, message in (
        ("manifest_sha256", "8" * 64, "manifest hash"),
        ("manifest_keys", ["wall/valid/00000/000001"], "exact keys"),
        ("horizons", {"1": {}, "5": {}}, "horizon set"),
    ):
        changed = {key: [dict(row) for row in rows] for key, rows in groups.items()}
        changed[("wall", "dinocular", 2)][0][field] = value
        with pytest.raises(CollectionError, match=message):
            _validate_open_loop_coverage(changed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evaluation_run_card_sha256", "0" * 64),
        ("training_run_card_sha256", "0" * 64),
        ("source_commit", "0" * 40),
        ("config_sha256", "0" * 64),
        ("container_sha256", "0" * 64),
        ("checkpoint_sha256", "0" * 64),
        ("depth_cache_manifest_sha256", "0" * 64),
        ("depth_native_contract_sha256", "0" * 64),
        ("depth_validation_sha256", "0" * 64),
    ],
)
def test_p4_collector_rejects_immutable_provenance_drift(field, value):
    groups = {}
    for arm_index, arm in enumerate(LOCKED_ARMS):
        for seed_index, seed in enumerate(LOCKED_SEEDS):
            token = "123456789"[arm_index * len(LOCKED_SEEDS) + seed_index]
            rows = []
            for episode in (0, 1):
                rows.append(
                    {
                        "run_id": f"p4-wall-{arm}-s{seed}",
                        "environment": "wall",
                        "arm": arm,
                        "seed": seed,
                        "episode": episode,
                        **_provenance_fields(arm=arm, token=token),
                        "manifest_keys": [f"wall/valid/{episode:05d}/000000"],
                        "horizons": {str(value): {} for value in (1, 5, 10)},
                    }
                )
            groups[("wall", arm, seed)] = rows
    changed = {key: [dict(row) for row in rows] for key, rows in groups.items()}
    changed[("wall", "dinocular", 2)][1][field] = value
    with pytest.raises(CollectionError, match="provenance drift"):
        _validate_open_loop_coverage(changed)


def test_existing_output_resume_rejects_immutable_provenance_drift(tmp_path):
    run_card_path = tmp_path / "evaluation.yaml"
    run_card_path.write_text("immutable: true\n", encoding="utf-8")
    card = {
        "run_card_sha256": "1" * 64,
        "training_run_card": {"file_sha256": "2" * 64},
        "source_commit": "f" * 40,
        "config_sha256": "3" * 64,
        "container": {"sha256": "4" * 64},
        "depth_inputs": {
            "producer_sha256": "5" * 64,
            "cache_manifest_sha256": "6" * 64,
            "native_contract_sha256": "7" * 64,
            "validation_sha256": "8" * 64,
            "checkpoint_sha256": "9" * 64,
        },
    }
    training_card = {"run_card_sha256": "a" * 64}
    expected = _result_provenance(
        card,
        training_card,
        run_card_path=run_card_path,
        checkpoint_sha256="b" * 64,
        manifest_sha256="c" * 64,
        slurm_job_id="12345",
    )
    resumed_row = dict(expected)
    resumed_row["slurm_job_id"] = "67890"
    _validate_existing_result_provenance(resumed_row, expected, requires_depth=True)
    for field, value in (
        ("evaluation_run_card_sha256", "0" * 64),
        ("training_run_card_sha256", "0" * 64),
        ("source_commit", "0" * 40),
        ("config_sha256", "0" * 64),
        ("container_sha256", "0" * 64),
        ("checkpoint_sha256", "0" * 64),
        ("manifest_sha256", "0" * 64),
        ("depth_producer_sha256", "0" * 64),
        ("depth_cache_manifest_sha256", "0" * 64),
        ("depth_native_contract_sha256", "0" * 64),
        ("depth_validation_sha256", "0" * 64),
    ):
        changed = dict(resumed_row)
        changed[field] = value
        with pytest.raises(EvaluationContractError, match="provenance differs"):
            _validate_existing_result_provenance(changed, expected, requires_depth=True)
    changed = dict(resumed_row)
    changed["slurm_job_id"] = "not-a-job"
    with pytest.raises(EvaluationContractError, match="slurm_job_id"):
        _validate_existing_result_provenance(changed, expected, requires_depth=True)


def test_pooled_nre_lower_point_decision_and_episode_bootstrap(tmp_path):
    paths = []
    for producer, multiplier in (
        ("da3_giant_video", 1.0),
        ("mapanything_recovered_framewise", 2.0),
    ):
        path = tmp_path / f"{producer}.jsonl"
        rows = [
            _result_row(producer, 0, 1 * multiplier, 2, 3 * multiplier, 6),
            _result_row(producer, 1, 9 * multiplier, 18, 1 * multiplier, 2),
        ]
        rows[1]["slurm_job_id"] = "12346"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        paths.append(path)
    output = tmp_path / "decision.json"
    tool = Path(__file__).resolve().parents[1] / "tools/collect_runs.py"
    subprocess.run(
        [
            "python3",
            str(tool),
            "producer-pilot",
            "--inputs",
            *(str(path) for path in paths),
            "--bootstrap",
            "10000",
            "--seed",
            "20260714",
            "--out",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    decision = json.loads(output.read_text())
    assert decision["winner"] == "da3_giant_video"
    assert decision["pooled"]["da3_giant_video"]["5"]["nre"] == pytest.approx(0.5)
    assert decision["paired_trajectory_bootstrap"]["unit"] == "held_out_episode"
    assert (
        decision["input_provenance"]["da3_giant_video"]["evaluation_run_card_sha256"]
        == "a" * 64
    )
    assert decision["input_provenance"]["da3_giant_video"]["slurm_job_ids"] == [
        "12345",
        "12346",
    ]
    rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
    rows[1]["config_sha256"] = "0" * 64
    paths[0].write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    completed = subprocess.run(
        [
            "python3",
            str(tool),
            "producer-pilot",
            "--inputs",
            *(str(path) for path in paths),
            "--bootstrap",
            "10000",
            "--seed",
            "20260714",
            "--out",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "provenance drift" in completed.stderr
