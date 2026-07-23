from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import lmdb
import numpy as np
import pytest
import torch
from torch import nn
import yaml
import zstandard

from datasets import depth_cache as depth_cache_module
from datasets.depth_cache import DepthCacheError, DepthCacheReader, EmpiricalDepthCacheReader
from depth_contract import DepthContractError, load_native_depth_contract, sha256_file
import empirical_depth_contract as empirical_contract_module
from empirical_depth_contract import (
    CAPSULE_RUNTIME_PATHS,
    CanonicalEmpiricalRuntimeResolver,
    EmpiricalRuntimePaths,
    EMPIRICAL_ADAPTER_ID,
    EMPIRICAL_ASSUMPTION,
    EMPIRICAL_CONTRACT_SCHEMA,
    EMPIRICAL_NON_EQUIVALENCE,
    EMPIRICAL_PROXY_SCALE,
    EmpiricalDepthContractError,
    apply_empirical_depth_adapter,
    load_depth_consumption_index,
    load_empirical_depth_contract,
    load_empirical_runtime_release,
    validate_empirical_provenance,
    validate_mapanything_receipt,
)
from models import dinocular_backbone
from models.dinocular import DinocularEncoder
from models.dinocular_backbone import BackendSpec
from tools import harness_common, make_manifests, submit_matrix, submit_p3_chain
from tools.collect_runs import _validate_open_loop_coverage
from tools.harness_common import (
    EVALUATION_IMMUTABLE_PROVENANCE_FIELDS,
    LOCKED_ARMS,
    LOCKED_ENVS,
    LOCKED_SEEDS,
    LOCKED_TARGETS,
    finalize_run_card,
    validate_empirical_depth_inputs,
)
from p3_completion import append_validation_record, validate_validation_records


CHECKPOINT_SHA = "decc7c73283bf46f66dbbedec6fb065ad0a943fb2ddb1c49317fafb0319f5dcc"
PRODUCER_SHA = "6996aa719531feb09f8dc858e66c2b703dc14393533f2d6dc073fd98937fe4e5"
MANIFEST_SHA = "9f35d303a5c604d5870ebdc6aedcefe9860bd0ab2763648c75b11e3b4f50b691"
RECEIPT_SHA = "ed3d63388a81581d20d560266a0d5e210b6672144fcb2cbb6a2e7ffef5736df9"
DATA_SHA = "68fe99e9f566eadcaa61e092b042962060fdc82c39ee8e71dcd7b9ef05247adb"
SOURCE_SHA = "ed6eecd62e455ffd35551074787f36b6c08039776b58a9dc6f22d17266ead6bd"
WIRE_SHA = "47a6d5944af9f3587ee7ef0b8154d157294e91db7e489c06b5b33bd3d9384ded"
CALIBRATION_SHA = "40eb3b62c3f77bee5d4399c46c2efba0672c05ed03ecbca2d91c5106f308115d"
MANIFEST_ID = "b87fe658-4731-4e8e-8c88-38f4fac6344c"


def _runtime_binding(prefix: str = "/empirical", mode: str = "canonical_host_v1") -> dict:
    return {
        "mode": mode,
        "paths": {
            "checkpoint": f"{prefix}/student.pth",
            "cache_directory": f"{prefix}/pusht.lmdb",
            "manifest": f"{prefix}/pusht.lmdb/manifest.json",
            "validation": f"{prefix}/validation.json",
            "data": f"{prefix}/pusht.lmdb/data.mdb",
            "empirical_contract": f"{prefix}/contract.json",
            "empirical_runtime_release": f"{prefix}/release.json",
        },
        "capsule_record_sha256": None,
        "deployment_acceptance_sha256": None,
    }


def _write_json(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(path)


def _contract_value(tmp_path: Path) -> dict:
    cache_dir = Path(
        "/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/"
        "data/depth_cache_mapanything_singleton/pusht.lmdb"
    )
    return {
        "schema": EMPIRICAL_CONTRACT_SCHEMA,
        "status": "complete",
        "empirical_use_allowed": True,
        "native_equivalence_claimed": False,
        "lossless_to_original_training_input": False,
        "assumption_tags": [EMPIRICAL_ASSUMPTION],
        "authority": {
            "route": "mapanything_primary_for_pusht",
            "authorization_basis": "owner_advisor_authorized_report_39",
            "execution_authority_granted": False,
        },
        "scientific_scope": {
            "valid_claim": "Exact frozen checkpoint driven by the accepted PushT cache through the declared empirical adapter.",
            "invalid_claims": [
                "reproduction_of_original_student_training_depth",
                "equality_to_unrecovered_original_mapanything_invocation",
                "checkpoint_native_input_equivalence",
                "lossless_recovery_of_raw_depth_z",
                "metric_depth_or_physical_unit_recovery",
                "native_neutral_or_rgb_only_equivalence",
                "mapanything_versus_depth_anything_3",
            ],
            "non_equivalence_statement": EMPIRICAL_NON_EQUIVALENCE,
        },
        "checkpoint": {
            "path": (
                "/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/"
                "checkpoints/dinov2_depthembed_dropout_fullpr.pth"
            ),
            "sha256": CHECKPOINT_SHA,
            "backend": "df2_dino_rope_convs_de",
            "factory": "DFormerv2_S",
            "checkpoint_key": "student",
            "state_prefix": "module.backbone.",
            "feature_key": "x_norm_patchtokens",
            "input_shape": [1, 224, 224],
        },
        "cache": {
            "environment": "pusht",
            "directory": str(cache_dir),
            "manifest_path": str(cache_dir / "manifest.json"),
            "manifest_sha256": MANIFEST_SHA,
            "manifest_id": MANIFEST_ID,
            "data_path": str(cache_dir / "data.mdb"),
            "data_sha256": DATA_SHA,
            "producer_sha256": PRODUCER_SHA,
            "source_index_sha256": SOURCE_SHA,
            "validation": {
                "path": (
                    "/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/"
                    "data/depth_cache_mapanything_singleton/validation.json"
                ),
                "sha256": RECEIPT_SHA,
                "schema": "dinocular-mapanything-cache-validation-v1",
                "state": "PASS",
            },
            "wire_format_sha256": WIRE_SHA,
        },
        "wire_semantics": {
            "stored_dtype": "little_endian_float16",
            "stored_shape": [224, 224],
            "stored_quantity": "clipped_normalized_later_mapanything_depth_z_proxy",
            "stored_range": [0.0, 1.0],
            "calibration": {
                "operation": "wire=clip((later_depth_z-lo)/(hi-lo),0,1)",
                "lo": 0.0,
                "hi": EMPIRICAL_PROXY_SCALE,
                "key_sha256": CALIBRATION_SHA,
                "scope": "pusht_training_only",
            },
            "geometry": "already_materialized_224x224_bilinear_depth",
            "irreversibilities": [
                "clipping",
                "float16_quantization_after_normalization",
                "bilinear_resampling_and_downsampling",
                "original_invocation_and_scale_not_recovered",
            ],
        },
        "adapter": {
            "id": EMPIRICAL_ADAPTER_ID,
            "informative_mode": {
                "decode": "zstd_then_little_endian_float16",
                "cast": "float16_to_float32",
                "reconstruction": "proxy_depth_z=wire_float32*1.5746406149864196",
                "spatial_operation": "identity_on_materialized_224x224",
                "additional_clipping": "none",
                "checkpoint_mean_std_normalization": "none",
                "invalid_policy": "fail_on_missing_nonfinite_or_out_of_range",
                "payload_presence_mask": "all_ones_only_after_payload_validation",
                "range_check": "exact_closed_interval_[0,1]_no_tolerance",
            },
            "zero_intervention_mode": {
                "requires_same_cache_identity_and_coverage": True,
                "boundary_value": 0.0,
                "boundary_mask": 0.0,
                "neutrality_claimed": False,
                "rgb_only_claimed": False,
                "label": "constant_zero_numeric_intervention_not_native_neutral",
            },
            "fallback_policy": {
                "isolated_frame_producer": "forbidden",
                "dynamic_reproduction": "forbidden",
                "alternate_cache": "forbidden",
                "missing_depth_to_zero": "forbidden",
                "rgb_only_substitution": "forbidden",
            },
        },
        "result_disclosure": {
            "required_assumption_tags": [EMPIRICAL_ASSUMPTION],
            "required_contract_sha256": True,
            "required_adapter_id": EMPIRICAL_ADAPTER_ID,
            "required_cache_identity_fields": True,
            "required_non_equivalence_statement": EMPIRICAL_NON_EQUIVALENCE,
        },
    }


def _loaded_empirical_index(
    tmp_path: Path,
    *,
    runtime_mode: str = "canonical_host_v1",
    contract_value: dict | None = None,
):
    contract_value = copy.deepcopy(contract_value or _contract_value(tmp_path))
    contract_path = tmp_path / "empirical-contract.json"
    contract_sha = _write_json(contract_path, contract_value)
    cache = contract_value["cache"]
    identities = {
        "checkpoint_sha256": contract_value["checkpoint"]["sha256"],
        "producer_sha256": cache["producer_sha256"],
        "manifest_sha256": cache["manifest_sha256"],
        "manifest_id": cache["manifest_id"],
        "validation_sha256": cache["validation"]["sha256"],
        "data_sha256": cache["data_sha256"],
        "source_index_sha256": cache["source_index_sha256"],
        "wire_format_sha256": cache["wire_format_sha256"],
    }
    canonical_paths = {
        "checkpoint": contract_value["checkpoint"]["path"],
        "cache_directory": cache["directory"],
        "manifest": cache["manifest_path"],
        "validation": cache["validation"]["path"],
        "data": cache["data_path"],
    }
    acceptance_path = tmp_path / "runtime-release-acceptance.json"
    acceptance_sha = _write_json(
        acceptance_path,
        {
            "schema": "dino-wm-empirical-runtime-release-acceptance-v1",
            "state": "INDEPENDENTLY_ACCEPTED",
            "bindings": {
                "empirical_contract_sha256": contract_sha,
                "identities": identities,
                "canonical_paths": canonical_paths,
                "runtime_mode": runtime_mode,
            },
        },
    )
    release = {
        "schema": "dino-wm-empirical-runtime-release-v1",
        "state": "INDEPENDENTLY_ACCEPTED",
        "empirical_contract_sha256": contract_sha,
        "identities": identities,
        "canonical_paths": canonical_paths,
        "runtime_mode": runtime_mode,
        "independent_acceptance": {
            "path": str(acceptance_path),
            "sha256": acceptance_sha,
        },
    }
    if runtime_mode == "capsule_v1":
        runtime_paths = dict(CAPSULE_RUNTIME_PATHS)
        capsule_path = tmp_path / "capsule.json"
        capsule_sha = _write_json(
            capsule_path,
            {
                "schema": "dino-wm-empirical-capsule-v1",
                "state": "ACCEPTED",
                "empirical_contract_sha256": contract_sha,
                "identities": identities,
                "runtime_paths": runtime_paths,
            },
        )
        deployment_path = tmp_path / "deployment.json"
        deployment_sha = _write_json(
            deployment_path,
            {
                "schema": "dino-wm-empirical-capsule-deployment-acceptance-v1",
                "state": "INDEPENDENTLY_ACCEPTED",
                "verdict": "PASS",
                "bindings": {
                    "empirical_contract_sha256": contract_sha,
                    "capsule_record_sha256": capsule_sha,
                    "runtime_paths": runtime_paths,
                },
            },
        )
        release["capsule"] = {
            "status": "READY",
            "path": str(capsule_path),
            "sha256": capsule_sha,
        }
        release["deployment_acceptance"] = {
            "status": "READY",
            "path": str(deployment_path),
            "sha256": deployment_sha,
        }
    release_path = tmp_path / "runtime-release.json"
    release_sha = _write_json(release_path, release)
    index_path = tmp_path / "depth-index.yaml"
    index_path.write_text(
        yaml.safe_dump(
            {
                "schema": "dino-wm-depth-consumption-index-v2",
                "defaults": None,
                "native_v1": {
                    "status": "BLOCKED_MISSING_ACCEPTED_ARTIFACT",
                    "index_path": None,
                    "index_sha256": None,
                    "delegated_environments": ["wall", "rope", "granular"],
                },
                "empirical_runtime_release": {
                    "status": "READY",
                    "release_path": str(release_path),
                    "release_sha256": release_sha,
                },
                "entries": {
                    "pusht/mapanything_recovered_framewise": {
                        "contract_kind": "empirical_lossy_cache",
                        "environment": "pusht",
                        "allowed_arms": ["dinocular", "dinocular_zerodepth"],
                        "contract_path": str(contract_path),
                        "contract_sha256": contract_sha,
                        "checkpoint_path": canonical_paths["checkpoint"],
                        "checkpoint_sha256": identities["checkpoint_sha256"],
                        "producer_sha256": identities["producer_sha256"],
                        "cache_dir": canonical_paths["cache_directory"],
                        "manifest_sha256": identities["manifest_sha256"],
                        "manifest_id": identities["manifest_id"],
                        "validation_path": canonical_paths["validation"],
                        "validation_sha256": identities["validation_sha256"],
                        "validation_schema": "dinocular-mapanything-cache-validation-v1",
                        "data_path": canonical_paths["data"],
                        "data_sha256": identities["data_sha256"],
                        "source_index_sha256": identities["source_index_sha256"],
                        "wire_format_sha256": identities["wire_format_sha256"],
                        "adapter_id": EMPIRICAL_ADAPTER_ID,
                        "assumption_tags": [EMPIRICAL_ASSUMPTION],
                        "execution_authority_granted": False,
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return load_depth_consumption_index(index_path)


def test_schema_separation_and_identity_mismatch_fail_closed(tmp_path: Path) -> None:
    value = _contract_value(tmp_path)
    path = tmp_path / "empirical.json"
    digest = _write_json(path, value)
    contract = load_empirical_depth_contract(path, digest, expected_checkpoint_sha256=CHECKPOINT_SHA)
    assert contract.wire_format_sha256 == WIRE_SHA
    with pytest.raises(DepthContractError, match="unsupported native depth schema"):
        load_native_depth_contract(path, digest)
    changed = copy.deepcopy(value)
    changed["cache"]["producer_sha256"] = "0" + PRODUCER_SHA[1:]
    changed_path = tmp_path / "changed.json"
    changed_digest = _write_json(changed_path, changed)
    with pytest.raises(EmpiricalDepthContractError, match="accepted PushT identity"):
        load_empirical_depth_contract(changed_path, changed_digest)


def test_discriminated_index_rejects_kind_schema_mismatch(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    contract_sha = _write_json(contract_path, _contract_value(tmp_path))
    contract_value = _contract_value(tmp_path)
    canonical_cache = contract_value["cache"]
    index = {
        "schema": "dino-wm-depth-consumption-index-v2",
        "defaults": None,
        "native_v1": {
            "status": "BLOCKED_MISSING_ACCEPTED_ARTIFACT",
            "index_path": "/missing/native-v1.yaml",
            "index_sha256": None,
            "delegated_environments": ["wall", "rope", "granular"],
        },
        "empirical_runtime_release": {
            "status": "BLOCKED_MISSING_ACCEPTED_ARTIFACT",
            "release_path": "/missing/release.json",
            "release_sha256": None,
        },
        "entries": {
            "pusht/mapanything": {
                "contract_kind": "empirical_lossy_cache",
                "contract_path": str(contract_path),
                "contract_sha256": contract_sha,
                "environment": "pusht",
                "cache_dir": canonical_cache["directory"],
                "validation_path": canonical_cache["validation"]["path"],
                "data_path": canonical_cache["data_path"],
                "checkpoint_path": contract_value["checkpoint"]["path"],
                "allowed_arms": ["dinocular", "dinocular_zerodepth"],
                "adapter_id": EMPIRICAL_ADAPTER_ID,
                "checkpoint_sha256": CHECKPOINT_SHA,
                "producer_sha256": PRODUCER_SHA,
                "manifest_sha256": MANIFEST_SHA,
                "manifest_id": MANIFEST_ID,
                "validation_sha256": RECEIPT_SHA,
                "validation_schema": "dinocular-mapanything-cache-validation-v1",
                "data_sha256": DATA_SHA,
                "source_index_sha256": SOURCE_SHA,
                "wire_format_sha256": WIRE_SHA,
                "assumption_tags": [EMPIRICAL_ASSUMPTION],
                "execution_authority_granted": False,
            }
        },
    }
    index_path = tmp_path / "index.yaml"
    index_path.write_text(yaml.safe_dump(index, sort_keys=False), encoding="utf-8")
    loaded = load_depth_consumption_index(index_path)
    assert loaded["entries"]["pusht/mapanything"]["contract_kind"] == "empirical_lossy_cache"
    index["entries"]["pusht/mapanything"]["contract_kind"] = "native"
    index_path.write_text(yaml.safe_dump(index, sort_keys=False), encoding="utf-8")
    with pytest.raises(EmpiricalDepthContractError, match="contract_kind/schema mismatch"):
        load_depth_consumption_index(index_path)


def test_adapter_golden_values_boundary_and_exact_zero_semantics() -> None:
    wire = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float16).reshape(1, 1, 1, 3)
    proxy, mask, diagnostics = apply_empirical_depth_adapter(wire, zero_intervention=False, require_224=False)
    expected = wire.float() * EMPIRICAL_PROXY_SCALE
    torch.testing.assert_close(proxy, expected, rtol=0, atol=0)
    torch.testing.assert_close(mask, torch.ones_like(proxy), rtol=0, atol=0)
    assert diagnostics == {"saturated_elements": 1, "element_count": 3, "saturation_fraction": 1 / 3}
    zero, zero_mask, zero_diagnostics = apply_empirical_depth_adapter(wire, zero_intervention=True, require_224=False)
    torch.testing.assert_close(zero, torch.zeros_like(proxy), rtol=0, atol=0)
    torch.testing.assert_close(zero_mask, torch.zeros_like(proxy), rtol=0, atol=0)
    assert zero_diagnostics["intervention"] == "exact_constant_zero_numeric"
    with pytest.raises(EmpiricalDepthContractError, match="224x224"):
        apply_empirical_depth_adapter(torch.zeros(1, 1, 223, 224), zero_intervention=False)
    with pytest.raises(EmpiricalDepthContractError, match="wire range"):
        apply_empirical_depth_adapter(torch.tensor([[[[1.1]]]]), zero_intervention=False, require_224=False)


def _provenance(contract_sha: str) -> dict:
    return {
        "contract_kind": "empirical_lossy_cache",
        "empirical_contract_sha256": contract_sha,
        "assumption_tags": [EMPIRICAL_ASSUMPTION],
        "adapter_id": EMPIRICAL_ADAPTER_ID,
        "adapter_mode": "proxy_depth_z",
        "producer_sha256": PRODUCER_SHA,
        "cache_manifest_sha256": MANIFEST_SHA,
        "manifest_id": MANIFEST_ID,
        "validation_sha256": RECEIPT_SHA,
        "validation_schema": "dinocular-mapanything-cache-validation-v1",
        "data_sha256": DATA_SHA,
        "source_index_sha256": SOURCE_SHA,
        "wire_format_sha256": WIRE_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "non_equivalence_statement": EMPIRICAL_NON_EQUIVALENCE,
        "execution_authority_granted": False,
        "neutrality_claimed": False,
        "rgb_only_claimed": False,
    }


def test_provenance_and_no_fallback_are_exact() -> None:
    contract_sha = "a" * 64
    provenance = _provenance(contract_sha)
    validate_empirical_provenance(provenance, expected_contract_sha256=contract_sha)
    validate_empirical_depth_inputs(provenance, arm="dinocular", environment="pusht")
    for field in ("assumption_tags", "wire_format_sha256", "non_equivalence_statement"):
        changed = copy.deepcopy(provenance)
        changed[field] = [] if field == "assumption_tags" else "changed"
        with pytest.raises((EmpiricalDepthContractError, Exception)):
            validate_empirical_provenance(changed, expected_contract_sha256=contract_sha)
    with pytest.raises(Exception, match="dino_pinned"):
        validate_empirical_depth_inputs(provenance, arm="dino_pinned", environment="pusht")
    with pytest.raises(Exception, match="PushT"):
        validate_empirical_depth_inputs(provenance, arm="dinocular", environment="wall")


def test_empirical_cache_reader_is_an_explicit_branch() -> None:
    assert EmpiricalDepthCacheReader is not DepthCacheReader
    assert EmpiricalDepthCacheReader.contract_kind == "empirical_lossy_cache"
    assert EmpiricalDepthCacheReader.validation_schema == "dinocular-mapanything-cache-validation-v1"


def test_exact_existing_mapanything_receipt_branch() -> None:
    manifest = {
        "manifest_id": MANIFEST_ID,
        "trajectory_count": 18706,
        "frame_count": 2339250,
        "wire_format": {
            "physical_key": "<split>/<episode:05d>/<frame:06d>",
            "dtype": "<f2",
            "shape": [224, 224],
            "order": "C",
            "compressor": "zstd",
            "compressor_level": 3,
            "map_size": 1 << 40,
            "normalization": "clip((depth_m-lo)/(hi-lo),0,1)",
            "inverted": False,
        },
        "calibration": {
            "scope": "pusht_training_only",
            "lo": 0.0,
            "hi": EMPIRICAL_PROXY_SCALE,
            "keys_sha256": CALIBRATION_SHA,
        },
    }
    receipt = {
        "schema": "dinocular-mapanything-cache-validation-v1",
        "state": "PASS",
        "manifest_id": MANIFEST_ID,
        "goal_gauge_gate": "IDENTICAL_SINGLETON_PATH_BY_CONSTRUCTION",
        "prefix_invariance_gate": "NOT_APPLICABLE_FRAMEWISE",
        "format_compatibility_gate": {
            "schema": "dinocular-depth-cache-v1",
            "producer_agnostic_fields_equal_to_da3": True,
            "wire_format": manifest["wire_format"],
        },
        "calibration_gate": {
            "scope": "pusht_training_only",
            "lo": 0.0,
            "hi": EMPIRICAL_PROXY_SCALE,
            "keys": 128,
            "validation_keys": 0,
            "keys_sha256": CALIBRATION_SHA,
        },
        "manifest_count_gate": {
            "dataset_frames": 2339250,
            "dataset_trajectories": 18706,
            "manifest_frames": 2339250,
            "manifest_trajectories": 18706,
        },
        "range_gate": {"actual_keys": 2339250, "expected_keys": 2339250},
        "independent_batch_equivalence_gate": {
            "frames": 3,
            "production_batch_size": 1,
            "max_absolute_error": 0.0,
            "threshold": 0.001,
        },
        "spot_recomputation_gate": {
            "frames": 3,
            "max_absolute_error_after_wire_decode": 0.000244140625,
            "threshold": 0.001,
        },
        "temporal_gate": {
            "state": "FAIL",
            "acceptance": "CHARACTERIZATION_ONLY_FRAMEWISE",
        },
    }
    validate_mapanything_receipt(receipt, manifest)
    changed = copy.deepcopy(receipt)
    changed["manifest_id"] = "other"
    with pytest.raises(EmpiricalDepthContractError, match="manifest"):
        validate_mapanything_receipt(changed, manifest)
    changed = copy.deepcopy(receipt)
    changed["schema"] = "dinocular-depth-cache-validation-v1"
    with pytest.raises(EmpiricalDepthContractError, match="schema"):
        validate_mapanything_receipt(changed, manifest)


class _EmpiricalTinyBackbone(nn.Module):
    def __init__(self, **_: object) -> None:
        super().__init__()
        self.rgb = nn.Conv2d(3, 8, 1)
        self.depth = nn.Conv2d(1, 8, 1)

    def forward_features(self, rgb: torch.Tensor, depth: torch.Tensor) -> dict[str, torch.Tensor]:
        value = torch.nn.functional.adaptive_avg_pool2d(
            self.rgb(rgb) + self.depth(depth), (7, 7)
        ).flatten(2).transpose(1, 2)
        return {"x_norm_patchtokens": value, "x_norm_clstoken": value[:, 0]}


def test_empirical_encoder_boundary_hook_has_only_frozen_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = BackendSpec(
        module=SimpleNamespace(Tiny=_EmpiricalTinyBackbone),
        factories={"Tiny": 8},
        output_kind="dino_feature_dict",
        checkpoint_policy="exact",
    )
    monkeypatch.setitem(dinocular_backbone.BACKEND_REGISTRY, "tiny_empirical", spec)
    source = _EmpiricalTinyBackbone()
    checkpoint = tmp_path / "student.pth"
    torch.save(
        {"student": {f"module.backbone.{key}": value for key, value in source.state_dict().items()}},
        checkpoint,
    )
    checkpoint_sha = sha256_file(checkpoint)
    fake_contract = SimpleNamespace(
        path=tmp_path / "empirical.json",
        sha256="a" * 64,
        manifest={"checkpoint": {"backend": "tiny_empirical", "factory": "Tiny", "checkpoint_key": "student", "state_prefix": "module.backbone.", "feature_key": "x_norm_patchtokens", "input_shape": [1, 224, 224]}},
        checkpoint_sha256=checkpoint_sha,
        checkpoint_path=checkpoint,
        producer_sha256=PRODUCER_SHA,
        wire_format_sha256=WIRE_SHA,
    )
    fake_resolver = SimpleNamespace(
        validate_open_path=lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr("models.dinocular.load_empirical_depth_contract", lambda *args, **kwargs: fake_contract)
    monkeypatch.setattr(
        "models.dinocular.load_empirical_runtime_release",
        lambda *args, **kwargs: SimpleNamespace(
            path=tmp_path / "release.json",
            sha256="b" * 64,
            mode="canonical_host_v1",
            resolver=fake_resolver,
        ),
    )
    encoder = DinocularEncoder(
        backend="tiny_empirical",
        factory="Tiny",
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=checkpoint_sha,
        checkpoint_key="student",
        state_prefix="module.backbone.",
        input_size=224,
        num_patches=49,
        emb_dim=8,
        empirical_depth_contract_path=str(tmp_path / "empirical.json"),
        empirical_depth_contract_sha256="a" * 64,
        empirical_runtime_release_path=str(tmp_path / "release.json"),
        empirical_runtime_release_sha256="b" * 64,
        depth_input_mode="empirical_lossy_cache_v1",
    )
    observed: list[tuple[torch.Tensor, torch.Tensor]] = []
    encoder.register_encoder_boundary_hook(
        lambda _module, depth, mask: observed.append((depth.detach().clone(), mask.detach().clone()))
    )
    wire = torch.full((1, 1, 224, 224), 0.5, dtype=torch.float16)
    encoder(torch.zeros(1, 3, 224, 224), wire, torch.ones_like(wire))
    torch.testing.assert_close(
        observed[0][0], wire.float() * EMPIRICAL_PROXY_SCALE, rtol=0, atol=0
    )
    torch.testing.assert_close(observed[0][1], torch.ones_like(observed[0][1]), rtol=0, atol=0)

    zero = copy.deepcopy(encoder)
    zero.empirical_zero_intervention = True
    observed_zero: list[tuple[torch.Tensor, torch.Tensor]] = []
    zero._encoder_boundary_hooks = [
        lambda _module, depth, mask: observed_zero.append((depth.detach().clone(), mask.detach().clone()))
    ]
    zero(torch.zeros(1, 3, 224, 224), wire, torch.ones_like(wire))
    torch.testing.assert_close(observed_zero[0][0], torch.zeros_like(observed_zero[0][0]), rtol=0, atol=0)
    torch.testing.assert_close(observed_zero[0][1], torch.zeros_like(observed_zero[0][1]), rtol=0, atol=0)


def test_locked_36_cells_and_targets_are_unchanged() -> None:
    cells = {(arm, env, seed) for arm in LOCKED_ARMS for env in LOCKED_ENVS for seed in LOCKED_SEEDS}
    assert len(cells) == 36
    assert LOCKED_ARMS == ("dino_pinned", "dinocular", "dinocular_zerodepth")
    assert LOCKED_ENVS == ("pusht", "wall", "rope", "granular")
    assert LOCKED_SEEDS == (1, 2, 3)
    assert LOCKED_TARGETS == {"pusht": 123858, "wall": 143910, "rope": 53500, "granular": 53500}


def test_real_study_spec_selects_mixed_v2_index_and_declares_blocked_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = yaml.safe_load((root / "conf/study_matrix.yaml").read_text(encoding="utf-8"))
    assert spec["contracts_index"].endswith("/depth_consumption_index_v2.yaml")
    index = yaml.safe_load(
        (root / "contracts/depth_consumption_index_v2.yaml").read_text(encoding="utf-8")
    )
    assert index["native_v1"]["status"] == "BLOCKED_MISSING_ACCEPTED_ARTIFACT"
    assert index["native_v1"]["index_sha256"] is None
    assert index["empirical_runtime_release"]["status"] == "BLOCKED_MISSING_ACCEPTED_ARTIFACT"
    assert index["empirical_runtime_release"]["release_sha256"] is None


def test_mixed_dispatch_uses_empirical_only_for_pusht_and_delegates_native_records(
    tmp_path: Path,
) -> None:
    native_records = {
        environment: {
            "producer": "mapanything_recovered_framewise",
            "producer_sha256": f"{index + 1}" * 64,
            "cache_dir": f"/native/{environment}",
            "cache_manifest_sha256": f"{index + 2}" * 64,
            "validation_path": f"/native/{environment}.json",
            "validation_sha256": f"{index + 3}" * 64,
            "native_contract_path": "/native/contract.json",
            "native_contract_sha256": "7" * 64,
            "checkpoint_sha256": CHECKPOINT_SHA,
        }
        for index, environment in enumerate(LOCKED_ENVS)
    }
    native_index = {
        "schema": "dino-wm-depth-contract-index-v1",
        "_test_depth_inputs": native_records,
    }
    mixed = harness_common.LoadedMixedContractIndex(
        empirical=_loaded_empirical_index(tmp_path),
        native_v1=native_index,
    )
    original = harness_common.depth_inputs

    def native_fixture(index: Mapping[str, object], producer: str, environment: str):
        if index is native_index:
            return copy.deepcopy(native_records[environment])
        return original(index, producer, environment)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(harness_common, "depth_inputs", native_fixture)
        pusht = original(mixed, "mapanything_recovered_framewise", "pusht")
        assert pusht["contract_kind"] == "empirical_lossy_cache"
        for environment in ("wall", "rope", "granular"):
            assert original(mixed, "mapanything_recovered_framewise", environment) == native_records[environment]


def test_empirical_contract_rejects_noncanonical_embedded_paths(tmp_path: Path) -> None:
    value = _contract_value(tmp_path)
    value["cache"]["directory"] = str(tmp_path / "pusht.lmdb")
    path = tmp_path / "empirical.json"
    digest = _write_json(path, value)
    with pytest.raises(EmpiricalDepthContractError, match="canonical"):
        load_empirical_depth_contract(path, digest)


def test_wire_range_is_exact_without_tolerance() -> None:
    for value in (-0.00005, 1.00005):
        with pytest.raises(EmpiricalDepthContractError, match="wire range"):
            apply_empirical_depth_adapter(
                torch.tensor([[[[value]]]], dtype=torch.float32),
                zero_intervention=False,
                require_224=False,
            )


def _empirical_validation_provenance(contract_sha: str) -> dict:
    return {
        "contract_kind": "empirical_lossy_cache",
        "empirical_contract_sha256": contract_sha,
        "assumption_tags": [EMPIRICAL_ASSUMPTION],
        "adapter_id": EMPIRICAL_ADAPTER_ID,
        "adapter_mode": "proxy_depth_z",
        "producer_sha256": PRODUCER_SHA,
        "cache_manifest_sha256": MANIFEST_SHA,
        "manifest_id": MANIFEST_ID,
        "validation_sha256": RECEIPT_SHA,
        "validation_schema": "dinocular-mapanything-cache-validation-v1",
        "data_sha256": DATA_SHA,
        "source_index_sha256": SOURCE_SHA,
        "wire_format_sha256": WIRE_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "non_equivalence_statement": EMPIRICAL_NON_EQUIVALENCE,
        "execution_authority_granted": False,
        "neutrality_claimed": False,
        "rgb_only_claimed": False,
    }


def test_empirical_validation_ledger_accepts_nested_provenance(tmp_path: Path) -> None:
    record = {
        "schema": "dino-wm.p3-heldout-loss.v1",
        "percent": 1,
        "global_step": 1,
        "target_steps": 100,
        "rounding_rule": "ceil(target_steps*percent/100)",
        "state_restored": True,
        "loss_numerator": 2.0,
        "element_count": 2,
        "mean_loss": 1.0,
        "slurm_job_id": "1",
        "source_commit": "f" * 40,
        "config_sha256": "1" * 64,
        "container_sha256": "2" * 64,
        "model_sha256": "3" * 64,
        "checkpoint_sha256": "4" * 64,
        "checkpoint_history_record_sha256": "5" * 64,
        "manifest_sha256": "6" * 64,
        "data_manifest_sha256": "7" * 64,
        "split_sha256": "8" * 64,
        "immutable_run_card_sha256": "9" * 64,
        "depth_producer_sha256": PRODUCER_SHA,
        "depth_cache_manifest_sha256": MANIFEST_SHA,
        "depth_native_contract_sha256": None,
        "depth_empirical_contract_sha256": "a" * 64,
        "depth_empirical_provenance": _empirical_validation_provenance("a" * 64),
        "depth_validation_sha256": RECEIPT_SHA,
        "depth_checkpoint_sha256": CHECKPOINT_SHA,
    }
    appended = append_validation_record(tmp_path / "validation.jsonl", record, target_steps=100)
    assert appended["depth_empirical_provenance"]["adapter_mode"] == "proxy_depth_z"
    with pytest.raises(Exception, match="adapter mode"):
        validate_validation_records(
            [appended],
            target_steps=100,
            immutable_run_card_sha256="9" * 64,
            manifest_sha256="6" * 64,
            expected_empirical_adapter_mode="exact_constant_zero_numeric",
        )


def _evaluation_row(arm: str, seed: int) -> dict:
    empirical = arm != "dino_pinned"
    row = {
        "schema": "dino-wm-open-loop-episode-errors-v1",
        "run_id": f"p4-pusht-{arm}-s{seed}",
        "environment": "pusht",
        "arm": arm,
        "seed": seed,
        "episode": 0,
        "manifest_keys": ["pusht/0"],
        "manifest_sha256": "8" * 64,
        "horizons": {
            str(horizon): {
                "model_squared_error": 1.0,
                "persistence_squared_error": 2.0,
                "element_count": 1,
            }
            for horizon in (1, 5, 10, 25)
        },
        "evaluation_run_card_sha256": f"{seed}" * 64,
        "evaluation_run_card_file_sha256": f"{seed + 1}" * 64,
        "training_run_card_sha256": f"{seed + 2}" * 64,
        "training_run_card_file_sha256": f"{seed + 3}" * 64,
        "source_commit": "f" * 40,
        "config_sha256": f"{seed + 4}" * 64,
        "container_sha256": "c" * 64,
        "checkpoint_sha256": f"{seed + 5}" * 64,
        "assumption_tags": [EMPIRICAL_ASSUMPTION] if empirical else [],
        "slurm_job_id": str(seed),
        "depth_producer_sha256": PRODUCER_SHA if empirical else None,
        "depth_cache_manifest_sha256": MANIFEST_SHA if empirical else None,
        "depth_native_contract_sha256": None,
        "depth_validation_sha256": RECEIPT_SHA if empirical else None,
        "depth_checkpoint_sha256": CHECKPOINT_SHA if empirical else None,
        "depth_contract_kind": "empirical_lossy_cache" if empirical else None,
        "depth_empirical_contract_sha256": "a" * 64 if empirical else None,
        "depth_adapter_id": EMPIRICAL_ADAPTER_ID if empirical else None,
        "depth_adapter_mode": (
            "exact_constant_zero_numeric" if arm == "dinocular_zerodepth" else "proxy_depth_z"
        ) if empirical else None,
        "depth_manifest_id": MANIFEST_ID if empirical else None,
        "depth_data_sha256": DATA_SHA if empirical else None,
        "depth_source_index_sha256": SOURCE_SHA if empirical else None,
        "depth_wire_format_sha256": WIRE_SHA if empirical else None,
        "depth_validation_schema": "dinocular-mapanything-cache-validation-v1" if empirical else None,
        "depth_non_equivalence_statement": EMPIRICAL_NON_EQUIVALENCE if empirical else None,
        "depth_neutrality_claimed": False if empirical else None,
        "depth_rgb_only_claimed": False if empirical else None,
        "depth_execution_authority_granted": False if empirical else None,
    }
    assert set(EVALUATION_IMMUTABLE_PROVENANCE_FIELDS).issubset(row)
    return row


def test_paired_collector_allows_only_arm_specific_empirical_adapter_mode() -> None:
    groups = {
        ("pusht", arm, seed): [_evaluation_row(arm, seed)]
        for arm in LOCKED_ARMS
        for seed in LOCKED_SEEDS
    }
    contracts = _validate_open_loop_coverage(groups)
    assert contracts["pusht"]["group_provenance"][("pusht", "dinocular", 1)][
        "depth_adapter_mode"
    ] == "proxy_depth_z"
    assert contracts["pusht"]["group_provenance"][("pusht", "dinocular_zerodepth", 1)][
        "depth_adapter_mode"
    ] == "exact_constant_zero_numeric"


def test_paired_collector_rejects_empirical_adapter_drift_in_any_seed() -> None:
    groups = {
        ("pusht", arm, seed): [_evaluation_row(arm, seed)]
        for arm in LOCKED_ARMS
        for seed in LOCKED_SEEDS
    }
    groups[("pusht", "dinocular", 1)][0]["depth_adapter_mode"] = (
        "exact_constant_zero_numeric"
    )
    with pytest.raises(Exception, match="arm-specific depth adapter provenance"):
        _validate_open_loop_coverage(groups)


def test_v2_dispatch_delegates_only_exact_declared_native_environments(
    tmp_path: Path,
) -> None:
    native = {
        "schema": "dino-wm-depth-contract-index-v1",
        "native_contract": {
            "path": "/native.json",
            "sha256": "7" * 64,
            "checkpoint_sha256": CHECKPOINT_SHA,
        },
        "producers": {
            "mapanything_recovered_framewise": {
                "producer_sha256": PRODUCER_SHA,
                "caches": {
                    environment: {
                        "cache_dir": f"/{environment}.lmdb",
                        "manifest_sha256": MANIFEST_SHA,
                        "validation_path": f"/{environment}.json",
                        "validation_sha256": RECEIPT_SHA,
                    }
                    for environment in ("wall", "rope", "granular", "pusht")
                },
            }
        },
    }
    mixed = harness_common.LoadedMixedContractIndex(
        empirical=_loaded_empirical_index(tmp_path),
        native_v1=native,
    )
    for environment in ("wall", "rope", "granular"):
        assert harness_common.depth_inputs(
            mixed, "mapanything_recovered_framewise", environment
        )["cache_dir"] == f"/{environment}.lmdb"
    assert harness_common.depth_inputs(
        mixed, "mapanything_recovered_framewise", "pusht"
    )["contract_kind"] == "empirical_lossy_cache"
    with pytest.raises(harness_common.HarnessError, match="absent"):
        harness_common.depth_inputs(mixed, "different-producer", "pusht")


def test_native_v1_loader_ignores_empirical_release_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native_path = tmp_path / "native-v1.yaml"
    native_path.write_text("native\n", encoding="utf-8")
    mixed_path = tmp_path / "mixed-v2.yaml"
    mixed_path.write_text(
        yaml.safe_dump(
            {
                "schema": "dino-wm-depth-consumption-index-v2",
                "native_v1": {
                    "status": "READY",
                    "index_path": str(native_path),
                    "index_sha256": "a" * 64,
                    "delegated_environments": ["wall", "rope", "granular"],
                },
                "empirical_runtime_release": {
                    "status": "BLOCKED_MISSING_ACCEPTED_ARTIFACT",
                    "release_path": None,
                    "release_sha256": None,
                },
                "entries": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(harness_common, "sha256_file", lambda _path: "a" * 64)
    monkeypatch.setattr(
        harness_common,
        "load_contract_index",
        lambda path: {
            "schema": harness_common.CONTRACT_INDEX_SCHEMA,
            "loaded_path": str(path),
        },
    )
    loaded = harness_common.load_native_contract_index(mixed_path)
    assert loaded["schema"] == harness_common.CONTRACT_INDEX_SCHEMA
    assert loaded["loaded_path"] == str(native_path)


def test_native_materialization_resolution_never_touches_empirical_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contracts_path = tmp_path / "mixed-v2.yaml"
    contracts_path.write_text("schema: dino-wm-depth-consumption-index-v2\n")
    spec = {"contracts_index": str(contracts_path)}
    native = {"schema": harness_common.CONTRACT_INDEX_SCHEMA}
    evidence = {
        "source_commit": "f" * 40,
        "source_file_sha256": {},
        "artifacts": {},
        "container": {},
    }
    monkeypatch.setattr(make_manifests, "load_yaml", lambda _path: spec)
    monkeypatch.setattr(make_manifests, "validate_spec", lambda _spec: None)
    monkeypatch.setattr(
        make_manifests, "load_native_contract_index", lambda _path: native
    )
    monkeypatch.setattr(
        make_manifests, "source_evidence", lambda _spec, _root: evidence
    )
    monkeypatch.setattr(
        make_manifests,
        "resolve_authorization_prerequisites",
        lambda *_args, **_kwargs: pytest.fail(
            "native resolution touched empirical prerequisites"
        ),
    )
    monkeypatch.setattr(
        make_manifests,
        "require_launch_authorization",
        lambda *_args, **_kwargs: pytest.fail(
            "native resolution touched empirical launch authority"
        ),
    )
    (tmp_path / "spec.yaml").write_text("spec\n", encoding="utf-8")
    args = SimpleNamespace(
        spec=tmp_path / "spec.yaml",
        contracts_index=contracts_path,
        local_code_root=tmp_path,
    )
    loaded_spec, loaded_contracts, loaded_evidence = (
        make_manifests._resolve_native_inputs(args)
    )
    assert loaded_spec == spec
    assert loaded_contracts == native
    assert loaded_evidence == evidence


def test_p2_materializes_only_native_legacy_cards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "conf/study_matrix.yaml").read_text(
            encoding="utf-8"
        )
    )
    evidence = {
        "source_commit": "f" * 40,
        "source_file_sha256": {},
        "artifacts": configured["artifacts"],
        "container": configured["container"],
    }
    monkeypatch.setattr(
        make_manifests,
        "_resolve_native_inputs",
        lambda _args: (configured, {"schema": "native"}, evidence),
    )
    monkeypatch.setattr(
        make_manifests,
        "_resolve_inputs",
        lambda _args: pytest.fail("P2 touched empirical candidate resolution"),
    )
    monkeypatch.setattr(
        make_manifests,
        "legacy_depth_inputs",
        lambda _index, producer, environment: {
            "producer": producer,
            "producer_sha256": "1" * 64,
            "cache_dir": f"/native/{environment}.lmdb",
            "cache_manifest_sha256": "2" * 64,
            "validation_path": f"/native/{environment}.json",
            "validation_sha256": "3" * 64,
            "native_contract_path": "/native/contract.json",
            "native_contract_sha256": "4" * 64,
            "checkpoint_sha256": "5" * 64,
        },
    )
    monkeypatch.setattr(
        make_manifests,
        "write_matrix",
        lambda _path, **values: values,
    )
    result = make_manifests.make_p2(
        SimpleNamespace(
            kind="geometry",
            producer=configured["p2a"]["producers"][0],
            out=tmp_path / "p2.yaml",
        )
    )
    cards = result["cards"]
    assert len(cards) == 12
    assert all(
        card["schema"] == harness_common.LEGACY_RUN_CARD_SCHEMA
        for card in cards
    )
    assert all("launch_authorization" not in card for card in cards)
    assert all(
        card.get("depth_inputs", {}).get("contract_kind") is None
        for card in cards
    )


def test_legacy_run_cards_bypass_only_empirical_authority() -> None:
    configured = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "conf/study_matrix.yaml").read_text(
            encoding="utf-8"
        )
    )
    base = make_manifests._base_card(
        configured,
        {
            "source_commit": "f" * 40,
            "source_file_sha256": {},
            "artifacts": configured["artifacts"],
            "container": configured["container"],
        },
        kind="p2-geometry",
        run_id="legacy-native",
        environment="pusht",
        arm="dino_pinned",
        seed=1,
        schema=harness_common.LEGACY_RUN_CARD_SCHEMA,
    )
    assert "launch_authorization_subject" not in base
    assert "launch_authorization" not in base
    assert "authorization_bindings" not in base
    harness_common.require_run_card_authorization(base, operation="submit")
    with pytest.raises(harness_common.HarnessError, match="legacy"):
        harness_common.require_run_card_authorization(
            dict(base, launch_authorization={}), operation="submit"
        )
    candidate = dict(
        base,
        schema=harness_common.RUN_CARD_SCHEMA,
        launch_authorization_subject=None,
        launch_authorization=None,
        authorization_bindings={},
    )
    with pytest.raises(harness_common.HarnessError, match="authority"):
        harness_common.require_run_card_authorization(candidate, operation="submit")


def test_legacy_dispatch_never_selects_empirical_v2_entry(tmp_path: Path) -> None:
    native = {
        "schema": "dino-wm-depth-contract-index-v1",
        "native_contract": {
            "path": "/native.json",
            "sha256": "7" * 64,
            "checkpoint_sha256": CHECKPOINT_SHA,
        },
        "producers": {
            "mapanything_recovered_framewise": {
                "producer_sha256": PRODUCER_SHA,
                "caches": {
                    "pusht": {
                        "cache_dir": "/native-pusht.lmdb",
                        "manifest_sha256": MANIFEST_SHA,
                        "validation_path": "/native-pusht.json",
                        "validation_sha256": RECEIPT_SHA,
                    }
                },
            }
        },
    }
    mixed = harness_common.LoadedMixedContractIndex(
        empirical=_loaded_empirical_index(tmp_path),
        native_v1=native,
    )
    result = harness_common.legacy_depth_inputs(
        mixed, "mapanything_recovered_framewise", "pusht"
    )
    assert result["cache_dir"] == "/native-pusht.lmdb"
    assert "contract_kind" not in result


def test_live_card_verification_discriminates_empirical_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    card = finalize_run_card(
        {
            "schema": "dino-wm-run-card-v1",
            "run_id": "empirical",
            "code_root": "/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/code/dino_wm",
            "source_commit": "f" * 40,
            "source_file_sha256": {},
            "artifacts": {},
            "container": {"path": "/tmp/container", "sha256": "1" * 64},
            "depth_inputs": {
                **_provenance("a" * 64),
                "empirical_contract_path": "/tmp/contract.json",
                "cache_dir": "/tmp/pusht.lmdb",
                "validation_path": "/tmp/validation.json",
                "data_path": "/tmp/pusht.lmdb/data.mdb",
                "empirical_runtime_release_path": "/tmp/release.json",
                "empirical_runtime_release_sha256": "b" * 64,
                "runtime_mode": "canonical_host_v1",
                "runtime_paths": _runtime_binding("/tmp")["paths"],
                "checkpoint_path": "/tmp/student.pth",
                "capsule_record_path": None,
                "capsule_record_sha256": None,
                "deployment_acceptance_path": None,
                "deployment_acceptance_sha256": None,
            },
        }
    )
    monkeypatch.setattr(submit_matrix.subprocess, "check_output", lambda *args, **kwargs: "f" * 40 + "\n" if "rev-parse" in args[0] else "")
    monkeypatch.setattr(submit_matrix, "require_real_marvin_path", lambda value, _label: value)
    monkeypatch.setattr(
        submit_matrix,
        "require_directory_no_alias",
        lambda path, _label: Path(path),
    )
    monkeypatch.setattr(
        submit_matrix,
        "require_regular_file_no_alias",
        lambda path, _label: Path(path),
    )
    monkeypatch.setattr(submit_matrix, "_file_hash", lambda path: {
        "/tmp/container": "1" * 64,
        "/tmp/pusht.lmdb/manifest.json": MANIFEST_SHA,
        "/tmp/validation.json": RECEIPT_SHA,
        "/tmp/contract.json": "a" * 64,
        "/tmp/release.json": "b" * 64,
        "/tmp/student.pth": CHECKPOINT_SHA,
    }[str(path)])
    submit_matrix.verify_live_card(card)


@pytest.mark.parametrize(
    "operation", ["materialize", "submit", "run_matrix", "evaluation", "chain"]
)
def test_false_authority_has_a_single_external_gate_api(operation: str) -> None:
    assert hasattr(harness_common, "require_launch_authorization")
    with pytest.raises(harness_common.HarnessError, match="authorization"):
        harness_common.require_launch_authorization(
            {
                "execution_authority_granted": False,
                "launch_authorization": None,
            },
            operation=operation,
        )


def _authorization_chain(
    tmp_path: Path,
    *,
    subject: str = "pusht-empirical-candidate-v1",
    source_commit: str = "f" * 40,
    source_files: Mapping[str, str] | None = None,
    contracts_index_sha: str = "2" * 64,
    empirical_contract_sha: str = "3" * 64,
    empirical_release_sha: str = "4" * 64,
) -> tuple[dict, dict, str]:
    source_files = dict(source_files or {"train.py": "1" * 64})

    records = {
        "source_release": {
            "schema": "dino-wm-source-release-v1",
            "state": "ACCEPTED",
            "authorization_subject": subject,
            "source_commit": source_commit,
            "source_file_sha256": source_files,
        },
    }
    source_path = tmp_path / "source-release.json"
    source_sha = _write_json(source_path, records["source_release"])
    records["empirical_implementation_acceptance"] = {
        "schema": "dino-wm-empirical-implementation-acceptance-v1",
        "state": "INDEPENDENTLY_ACCEPTED",
        "verdict": "PASS",
        "authorization_subject": subject,
        "bindings": {
            "contracts_index_sha256": contracts_index_sha,
            "empirical_contract_sha256": empirical_contract_sha,
            "source_release_sha256": source_sha,
        },
    }
    implementation_path = tmp_path / "implementation-acceptance.json"
    implementation_sha = _write_json(
        implementation_path, records["empirical_implementation_acceptance"]
    )
    records["immutable_execution_acceptance"] = {
        "schema": "dino-wm-immutable-execution-acceptance-v1",
        "state": "INDEPENDENTLY_ACCEPTED",
        "verdict": "PASS",
        "authorization_subject": subject,
        "bindings": {
            "empirical_runtime_release_sha256": empirical_release_sha,
            "source_release_sha256": source_sha,
            "empirical_implementation_acceptance_sha256": implementation_sha,
        },
    }
    execution_path = tmp_path / "execution-acceptance.json"
    execution_sha = _write_json(
        execution_path, records["immutable_execution_acceptance"]
    )
    records["immutable_probe_acceptance"] = {
        "schema": "dino-wm-immutable-probe-acceptance-v1",
        "state": "INDEPENDENTLY_ACCEPTED",
        "verdict": "PASS",
        "authorization_subject": subject,
        "bindings": {
            "empirical_runtime_release_sha256": empirical_release_sha,
            "immutable_execution_acceptance_sha256": execution_sha,
        },
    }
    probe_path = tmp_path / "probe-acceptance.json"
    probe_sha = _write_json(probe_path, records["immutable_probe_acceptance"])
    spec = {
        "launch_authorization_subject": subject,
        "authorization_prerequisites": {
            "source_release": {
                "status": "READY",
                "path": str(source_path),
                "sha256": source_sha,
            },
            "empirical_implementation_acceptance": {
                "status": "READY",
                "path": str(implementation_path),
                "sha256": implementation_sha,
            },
            "immutable_execution_acceptance": {
                "status": "READY",
                "path": str(execution_path),
                "sha256": execution_sha,
            },
            "immutable_probe_acceptance": {
                "status": "READY",
                "path": str(probe_path),
                "sha256": probe_sha,
            },
        },
    }
    bindings = harness_common.resolve_authorization_prerequisites(
        spec,
        contracts_index_sha256=contracts_index_sha,
        empirical_contract_sha256=empirical_contract_sha,
        empirical_runtime_release_sha256=empirical_release_sha,
        source_commit=source_commit,
        source_file_sha256=source_files,
    )
    authorization = {
        "schema": "dino-wm-launch-authorization-v1",
        "state": "AUTHORIZED",
        "decision": "EXPLICIT_LAUNCH_AUTHORIZED",
        "execution_authority_granted": True,
        "authorization_subject": subject,
        "authorized_operations": sorted(harness_common.LAUNCH_OPERATIONS),
        "bindings": bindings,
    }
    authorization_path = tmp_path / "launch-authorization.json"
    authorization_sha = _write_json(authorization_path, authorization)
    return spec, bindings, authorization_sha


def test_launch_authorization_rejects_nonclosed_binding_set(tmp_path: Path) -> None:
    spec, bindings, authorization_sha = _authorization_chain(tmp_path)
    authorization_path = tmp_path / "launch-authorization.json"
    changed = json.loads(authorization_path.read_text(encoding="utf-8"))
    changed["bindings"]["unreviewed_identity_sha256"] = "a" * 64
    changed_sha = _write_json(authorization_path, changed)
    candidate = {
        **spec,
        "launch_authorization": {
            "status": "AUTHORIZED",
            "path": str(authorization_path),
            "sha256": changed_sha,
        },
        "authorization_bindings": {
            **bindings,
            "unreviewed_identity_sha256": "a" * 64,
        },
    }
    with pytest.raises(harness_common.HarnessError, match="binding set"):
        harness_common.require_launch_authorization(candidate, operation="materialize")


def test_manifest_resolution_uses_external_authorization_without_spec_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, bindings, authorization_sha = _authorization_chain(tmp_path)
    authorization_path = tmp_path / "launch-authorization.json"
    spec_path = tmp_path / "study.yaml"
    index_path = tmp_path / "index.yaml"
    spec.update({"contracts_index": str(index_path)})
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    index_path.write_text(
        yaml.safe_dump(
            {
                "entries": {
                    "pusht/mapanything_recovered_framewise": {
                        "contract_sha256": bindings["empirical_contract_sha256"]
                    }
                },
                "empirical_runtime_release": {
                    "release_sha256": bindings[
                        "empirical_runtime_release_sha256"
                    ]
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(make_manifests, "validate_spec", lambda _spec: None)
    monkeypatch.setattr(
        make_manifests,
        "source_evidence",
        lambda _spec, _root: {
            "source_commit": bindings["source_commit"],
            "source_file_sha256": {"train.py": "1" * 64},
            "artifacts": {},
            "container": {},
        },
    )
    monkeypatch.setattr(make_manifests, "load_contract_index", lambda _path: {})
    monkeypatch.setattr(
        make_manifests,
        "sha256_file",
        lambda _path: bindings["contracts_index_sha256"],
    )
    resolved, _contracts, _evidence = make_manifests._resolve_inputs(
        SimpleNamespace(
            spec=spec_path,
            contracts_index=index_path,
            local_code_root=tmp_path,
            launch_authorization=authorization_path,
            launch_authorization_sha256=authorization_sha,
        )
    )
    assert resolved["authorization_bindings"] == bindings
    assert "study_spec_sha256" not in bindings
    assert resolved["launch_authorization"]["path"] == str(authorization_path)


@pytest.mark.parametrize("wire_value", [-5e-5, 1.001])
def test_empirical_reader_rejects_any_out_of_range_decoded_wire_value(
    wire_value: float,
) -> None:
    reader = object.__new__(EmpiricalDepthCacheReader)
    value = np.full((224, 224), wire_value, dtype="<f2")
    with pytest.raises(DepthCacheError, match="approved wire range"):
        reader._validate_wire_range(value, "valid/00000/000000")


def test_empirical_reader_uses_release_identity_without_rehashing_data_or_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source = source_root / "pusht_noise" / "val" / "00000.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"accepted-source-bytes")
    cache_dir = tmp_path / "pusht.lmdb"
    database = lmdb.open(str(cache_dir), map_size=64 << 20)
    compressor = zstandard.ZstdCompressor(level=3)
    with database.begin(write=True) as transaction:
        value = np.full((224, 224), 0.5, dtype="<f2")
        transaction.put(b"valid/00000/000000", compressor.compress(value.tobytes()))
    database.sync(True)
    database.close()
    producer = {"name": "accepted-test-producer", "version": "1"}
    wire = {
        "physical_key": "<split>/<episode:05d>/<frame:06d>",
        "dtype": "<f2",
        "shape": [224, 224],
        "order": "C",
        "compressor": "zstd",
        "compressor_level": 3,
        "map_size": 1 << 40,
    }
    source_sha = sha256_file(source)
    source_index = [
        {
            "key": "valid/00000",
            "frames": 1,
            "source": "pusht_noise/val/00000.mp4",
            "sha256": source_sha,
        }
    ]
    manifest = {
        "schema": "dinocular-depth-cache-v1",
        "environment": "pusht",
        "manifest_id": "tiny",
        "trajectory_count": 1,
        "frame_count": 1,
        "wire_format": wire,
        "closed_before_hash": True,
        "producer": producer,
        "source_index_sha256": sha256_file(source),
        "data_mdb_sha256": sha256_file(cache_dir / "data.mdb"),
        "calibration": {
            "scope": "pusht_training_only",
            "lo": 0.0,
            "hi": EMPIRICAL_PROXY_SCALE,
            "keys_sha256": CALIBRATION_SHA,
        },
        "trajectories": [
            {
                "trajectory_key": "valid/00000",
                "source_path": "pusht_noise/val/00000.mp4",
                "source_video_sha256": source_sha,
                "ordered_frame_count": 1,
                "ordered_output_keys": ["valid/00000/000000"],
            }
        ],
    }
    manifest["source_index_sha256"] = depth_cache_module.sha256_bytes(
        depth_cache_module.canonical_json_bytes(source_index)
    )
    manifest_path = cache_dir / "manifest.json"
    manifest_sha = _write_json(manifest_path, manifest)
    validation_path = tmp_path / "validation.json"
    validation_sha = _write_json(validation_path, {"schema": "tiny"})
    fake_contract = SimpleNamespace(
        cache_directory=cache_dir,
        manifest_path=manifest_path,
        validation_path=validation_path,
        data_path=cache_dir / "data.mdb",
        manifest_sha256=manifest_sha,
        validation_sha256=validation_sha,
        manifest_id="tiny",
        wire_format_sha256=depth_cache_module.sha256_bytes(
            depth_cache_module.canonical_json_bytes(wire)
        ),
        producer_sha256=depth_cache_module.sha256_bytes(
            depth_cache_module.canonical_json_bytes(producer)
        ),
        source_index_sha256=manifest["source_index_sha256"],
        data_sha256=manifest["data_mdb_sha256"],
        manifest={
            "wire_semantics": {
                "calibration": {
                    "scope": "pusht_training_only",
                    "lo": 0.0,
                    "hi": EMPIRICAL_PROXY_SCALE,
                    "key_sha256": CALIBRATION_SHA,
                }
            }
        },
    )
    monkeypatch.setattr(depth_cache_module, "load_empirical_depth_contract", lambda *args, **kwargs: fake_contract)
    monkeypatch.setattr(
        depth_cache_module,
        "load_empirical_runtime_release",
        lambda *args, **kwargs: SimpleNamespace(
            resolver=SimpleNamespace(
                validate_open_path=lambda *_args, **_kwargs: None
            )
        ),
    )
    monkeypatch.setattr(depth_cache_module, "validate_mapanything_receipt", lambda *args, **kwargs: None)
    original_sha256_file = depth_cache_module.sha256_file

    def bounded_hash(path: Path) -> str:
        candidate = Path(path)
        if candidate in {cache_dir / "data.mdb", source}:
            raise AssertionError(f"ordinary reader rehashed accepted bulk object: {candidate}")
        return original_sha256_file(candidate)

    monkeypatch.setattr(depth_cache_module, "sha256_file", bounded_hash)
    reader = EmpiricalDepthCacheReader(
        environment="pusht",
        source_root=source_root,
        empirical_contract_path=tmp_path / "contract.json",
        empirical_contract_sha256="a" * 64,
        empirical_runtime_release_path=tmp_path / "release.json",
        empirical_runtime_release_sha256="b" * 64,
    )
    reader.assert_dataset_coverage([("valid", 0, 1)])
    depth, mask = reader.read(split="valid", episode=0, frames=[0])
    assert float(depth[0, 0, 0]) == 0.5
    assert torch.all(mask == 1)


def test_real_make_training_materializes_36_mixed_cards_with_arm_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[1]
    spec = yaml.safe_load((root / "conf/study_matrix.yaml").read_text(encoding="utf-8"))
    evidence = {
        "source_commit": "f" * 40,
        "source_file_sha256": {"train.py": "1" * 64},
        "artifacts": spec["artifacts"],
        "container": spec["container"],
    }
    native_entries = {}
    for index, environment in enumerate(LOCKED_ENVS, 1):
        native_entries[environment] = {
            "producer": "mapanything_recovered_framewise",
            "producer_sha256": f"{index}" * 64,
            "cache_dir": f"{spec['study_root']}/native/{environment}.lmdb",
            "cache_manifest_sha256": f"{index + 1}" * 64,
            "validation_path": f"{spec['study_root']}/native/{environment}.validation.json",
            "validation_sha256": f"{index + 2}" * 64,
            "native_contract_path": f"{spec['study_root']}/manifests/native.json",
            "native_contract_sha256": "7" * 64,
            "checkpoint_sha256": CHECKPOINT_SHA,
        }
    native_index = {"schema": "dino-wm-depth-contract-index-v1", "_test_entries": native_entries}
    mixed_index = harness_common.LoadedMixedContractIndex(
        empirical=_loaded_empirical_index(tmp_path),
        native_v1=native_index,
    )
    original_depth_inputs = harness_common.depth_inputs

    def dispatch(index: Mapping[str, object], producer: str, environment: str):
        if index is native_index:
            return copy.deepcopy(native_entries[environment])
        return original_depth_inputs(index, producer, environment)

    monkeypatch.setattr(harness_common, "depth_inputs", dispatch)
    monkeypatch.setattr(make_manifests, "depth_inputs", dispatch)
    monkeypatch.setattr(make_manifests, "_resolve_inputs", lambda _args: (spec, mixed_index, evidence))
    monkeypatch.setattr(make_manifests, "_load_winner", lambda *_args: "mapanything_recovered_framewise")
    monkeypatch.setattr(
        make_manifests,
        "_heldout_manifest_record",
        lambda _directory, environment, _commit, target: {
            "path": f"{spec['study_root']}/heldout/{environment}.jsonl",
            "sha256": "1" * 64,
            "metadata_path": f"{spec['study_root']}/heldout/{environment}.meta.json",
            "metadata_sha256": "2" * 64,
            "data_manifest_sha256": "3" * 64,
            "split_sha256": "4" * 64,
            "selection": "all_validation_examples",
            "entry_count": 1,
            "target_steps": target,
            "rounding_rule": "ceil(target_steps*percent/100)",
        },
    )
    monkeypatch.setattr(
        make_manifests,
        "_segment_record",
        lambda _spec, _evidence, _rates, arm, environment, target_steps: {
            "derived_segment_steps": 1000,
            "timing_source_commit": "f" * 40,
            "arm": arm,
            "environment": environment,
            "target_steps": target_steps,
        },
    )
    captured: dict[str, object] = {}

    def capture_matrix(_path: Path, *, kind: str, cards: list[Mapping[str, object]], source_commit: str):
        captured.update(kind=kind, cards=cards, source_commit=source_commit)
        return {"kind": kind, "card_count": len(cards)}

    monkeypatch.setattr(make_manifests, "write_matrix", capture_matrix)
    (tmp_path / "decision.json").write_text("{}\n", encoding="utf-8")
    result = make_manifests.make_training(
        SimpleNamespace(
            encoders=",".join(LOCKED_ARMS),
            envs=",".join(LOCKED_ENVS),
            seeds="1,2,3",
            batch_size=32,
            predictor_lr=0.00005,
            decoder="off",
            target_steps="pusht=123858,wall=143910,rope=53500,granular=53500",
            frameskips="pusht=5,wall=5,rope=1,granular=1",
            producer_decision=tmp_path / "decision.json",
            heldout_manifests_dir=tmp_path,
            rates=tmp_path / "rates.json",
            out=tmp_path / "matrix.yaml",
        )
    )
    assert result["card_count"] == 36
    cards = list(captured["cards"])
    assert len(cards) == 36
    for card in cards:
        if card["arm"] == "dino_pinned":
            assert "depth_inputs" not in card
        elif card["environment"] == "pusht":
            assert card["depth_inputs"]["contract_kind"] == "empirical_lossy_cache"
            expected_config = (
                "encoder=dinocular_zerodepth_pusht_empirical"
                if card["arm"] == "dinocular_zerodepth"
                else "encoder=dinocular_pusht_empirical"
            )
            assert expected_config in card["overrides"]
        else:
            assert "contract_kind" not in card["depth_inputs"]
            assert f"encoder={card['arm']}" in card["overrides"]


def test_loader_produced_accepted_evidence_traverses_full_empirical_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source_path = source_root / "pusht_noise" / "val" / "00000.mp4"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"accepted-source")

    cache_dir = tmp_path / "pusht.lmdb"
    database = lmdb.open(str(cache_dir), map_size=64 << 20)
    compressor = zstandard.ZstdCompressor(level=3)
    with database.begin(write=True) as transaction:
        wire_value = np.full((224, 224), 0.5, dtype="<f2")
        transaction.put(
            b"valid/00000/000000", compressor.compress(wire_value.tobytes())
        )
    database.sync(True)
    database.close()

    producer = {"name": "accepted-route-producer", "version": "1"}
    wire_format = {
        "physical_key": "<split>/<episode:05d>/<frame:06d>",
        "dtype": "<f2",
        "shape": [224, 224],
        "order": "C",
        "compressor": "zstd",
        "compressor_level": 3,
        "map_size": 1 << 40,
    }
    source_index = [
        {
            "key": "valid/00000",
            "frames": 1,
            "source": "pusht_noise/val/00000.mp4",
            "sha256": sha256_file(source_path),
        }
    ]
    identities = {
        "producer_sha256": depth_cache_module.sha256_bytes(
            depth_cache_module.canonical_json_bytes(producer)
        ),
        "source_index_sha256": depth_cache_module.sha256_bytes(
            depth_cache_module.canonical_json_bytes(source_index)
        ),
        "wire_format_sha256": depth_cache_module.sha256_bytes(
            depth_cache_module.canonical_json_bytes(wire_format)
        ),
        "data_sha256": sha256_file(cache_dir / "data.mdb"),
        "calibration_key_sha256": depth_cache_module.sha256_bytes(
            depth_cache_module.canonical_json_bytes(["valid/00000/000000"])
        ),
    }
    manifest_id = "temporary-byte-valid-accepted-route"
    manifest = {
        "schema": "dinocular-depth-cache-v1",
        "environment": "pusht",
        "manifest_id": manifest_id,
        "trajectory_count": 1,
        "frame_count": 1,
        "wire_format": wire_format,
        "closed_before_hash": True,
        "producer": producer,
        "source_index_sha256": identities["source_index_sha256"],
        "data_mdb_sha256": identities["data_sha256"],
        "calibration": {
            "scope": "pusht_training_only",
            "lo": 0.0,
            "hi": EMPIRICAL_PROXY_SCALE,
            "keys_sha256": identities["calibration_key_sha256"],
        },
        "trajectories": [
            {
                "trajectory_key": "valid/00000",
                "source_path": "pusht_noise/val/00000.mp4",
                "source_video_sha256": sha256_file(source_path),
                "ordered_frame_count": 1,
                "ordered_output_keys": ["valid/00000/000000"],
            }
        ],
    }
    manifest_path = cache_dir / "manifest.json"
    identities["manifest_sha256"] = _write_json(manifest_path, manifest)
    validation = {
        "schema": "dinocular-mapanything-cache-validation-v1",
        "state": "PASS",
        "manifest_id": manifest_id,
        "goal_gauge_gate": "IDENTICAL_SINGLETON_PATH_BY_CONSTRUCTION",
        "prefix_invariance_gate": "NOT_APPLICABLE_FRAMEWISE",
        "format_compatibility_gate": {
            "schema": "dinocular-depth-cache-v1",
            "producer_agnostic_fields_equal_to_da3": True,
            "wire_format": wire_format,
        },
        "calibration_gate": {
            "scope": "pusht_training_only",
            "lo": 0.0,
            "hi": EMPIRICAL_PROXY_SCALE,
            "keys": 128,
            "validation_keys": 0,
            "keys_sha256": identities["calibration_key_sha256"],
        },
        "manifest_count_gate": {
            "dataset_frames": 1,
            "dataset_trajectories": 1,
            "manifest_frames": 1,
            "manifest_trajectories": 1,
        },
        "range_gate": {"actual_keys": 1, "expected_keys": 1},
        "independent_batch_equivalence_gate": {
            "frames": 3,
            "production_batch_size": 1,
            "max_absolute_error": 0.0,
            "threshold": 0.001,
        },
        "spot_recomputation_gate": {
            "frames": 3,
            "max_absolute_error_after_wire_decode": 0.000244140625,
            "threshold": 0.001,
        },
        "temporal_gate": {
            "state": "FAIL",
            "acceptance": "CHARACTERIZATION_ONLY_FRAMEWISE",
        },
    }
    validation_path = tmp_path / "validation.json"
    identities["validation_sha256"] = _write_json(validation_path, validation)

    source_backbone, _backend_spec = dinocular_backbone.build_backbone(
        "df2_dino_rope_convs_de", "DFormerv2_S"
    )
    checkpoint_path = tmp_path / "student.pth"
    torch.save(
        {
            "student": {
                f"module.backbone.{key}": value
                for key, value in source_backbone.state_dict().items()
            }
        },
        checkpoint_path,
    )
    identities["checkpoint_sha256"] = sha256_file(checkpoint_path)
    identities["manifest_id"] = manifest_id

    canonical_paths = {
        "checkpoint": str(checkpoint_path),
        "cache_directory": str(cache_dir),
        "manifest": str(manifest_path),
        "validation": str(validation_path),
        "data": str(cache_dir / "data.mdb"),
    }
    monkeypatch.setattr(
        empirical_contract_module, "ACCEPTED_IDENTITIES", dict(identities)
    )
    monkeypatch.setattr(
        empirical_contract_module, "ACCEPTED_CANONICAL_PATHS", canonical_paths
    )
    contract_value = _contract_value(tmp_path)
    contract_value["checkpoint"].update(
        path=str(checkpoint_path), sha256=identities["checkpoint_sha256"]
    )
    contract_value["cache"].update(
        directory=str(cache_dir),
        manifest_path=str(manifest_path),
        manifest_sha256=identities["manifest_sha256"],
        manifest_id=manifest_id,
        data_path=str(cache_dir / "data.mdb"),
        data_sha256=identities["data_sha256"],
        producer_sha256=identities["producer_sha256"],
        source_index_sha256=identities["source_index_sha256"],
        wire_format_sha256=identities["wire_format_sha256"],
    )
    contract_value["cache"]["validation"].update(
        path=str(validation_path), sha256=identities["validation_sha256"]
    )
    contract_value["wire_semantics"]["calibration"]["key_sha256"] = identities[
        "calibration_key_sha256"
    ]
    loaded = _loaded_empirical_index(tmp_path, contract_value=contract_value)
    native_entries = {
        environment: {
            "producer": "mapanything_recovered_framewise",
            "producer_sha256": f"{index}" * 64,
            "cache_dir": str(tmp_path / "native" / f"{environment}.lmdb"),
            "cache_manifest_sha256": f"{index + 1}" * 64,
            "validation_path": str(
                tmp_path / "native" / f"{environment}.validation.json"
            ),
            "validation_sha256": f"{index + 2}" * 64,
            "native_contract_path": str(tmp_path / "native-contract.json"),
            "native_contract_sha256": "7" * 64,
            "checkpoint_sha256": identities["checkpoint_sha256"],
        }
        for index, environment in enumerate(LOCKED_ENVS, 1)
    }
    native_index = {
        "schema": "dino-wm-depth-contract-index-v1",
        "_test_entries": native_entries,
    }
    mixed_index = harness_common.LoadedMixedContractIndex(
        empirical=loaded, native_v1=native_index
    )
    original_depth_inputs = harness_common.depth_inputs

    def dispatch_depth_inputs(index, producer_name, environment):
        if index is native_index:
            return copy.deepcopy(native_entries[environment])
        return original_depth_inputs(index, producer_name, environment)

    monkeypatch.setattr(make_manifests, "depth_inputs", dispatch_depth_inputs)

    code_root = tmp_path / "code"
    code_root.mkdir()
    source_file = code_root / "route_source.py"
    source_file.write_text("ROUTE = 'accepted'\n", encoding="utf-8")
    dinov2_path = tmp_path / "dinov2.pth"
    dinov2_path.write_bytes(b"dinov2")
    container_path = tmp_path / "container.sif"
    container_path.write_bytes(b"container")
    source_files = {"route_source.py": sha256_file(source_file)}
    index_path = tmp_path / "depth-index.yaml"
    contract_path = tmp_path / "empirical-contract.json"
    release_path = tmp_path / "runtime-release.json"
    authorization_spec, bindings, authorization_sha = _authorization_chain(
        tmp_path,
        source_files=source_files,
        contracts_index_sha=sha256_file(index_path),
        empirical_contract_sha=sha256_file(contract_path),
        empirical_release_sha=sha256_file(release_path),
    )
    configured = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "conf/study_matrix.yaml").read_text(
            encoding="utf-8"
        )
    )
    configured.update(authorization_spec)
    configured.update(
        {
            "study_root": str(tmp_path),
            "code_root": str(code_root),
            "artifacts": {
                "dinov2": {
                    "path": str(dinov2_path),
                    "sha256": sha256_file(dinov2_path),
                },
                "dinocular_student": {
                    "path": str(checkpoint_path),
                    "sha256": identities["checkpoint_sha256"],
                },
            },
            "container": {
                "path": str(container_path),
                "sha256": sha256_file(container_path),
            },
            "launch_authorization": {
                "status": "AUTHORIZED",
                "path": str(tmp_path / "launch-authorization.json"),
                "sha256": authorization_sha,
            },
            "authorization_bindings": bindings,
        }
    )
    evidence = {
        "source_commit": "f" * 40,
        "source_file_sha256": source_files,
        "artifacts": configured["artifacts"],
        "container": configured["container"],
    }
    timing_path = tmp_path / "timing.json"
    _write_json(
        timing_path,
        {
            "schema": harness_common.TIMING_SUMMARY_SCHEMA,
            "state": "PASS",
            "source_commit": evidence["source_commit"],
            "matrix_sha256": "0" * 64,
            "rates": {
                f"{arm}/{environment}": {
                    "state": "PASS",
                    "optimizer_steps_per_second": 1.0,
                }
                for arm in LOCKED_ARMS
                for environment in LOCKED_ENVS
            },
        },
    )
    producer_decision = tmp_path / "decision.json"
    _write_json(producer_decision, {"winner": "mapanything_recovered_framewise"})
    heldout_path = tmp_path / "heldout_pusht.jsonl"
    heldout_path.write_text("{}\n", encoding="utf-8")
    heldout_metadata = tmp_path / "heldout_pusht.meta.json"
    _write_json(heldout_metadata, {"state": "PASS"})
    heldout = {
        "pusht": {
            "path": str(heldout_path),
            "sha256": sha256_file(heldout_path),
            "metadata_path": str(heldout_metadata),
            "metadata_sha256": sha256_file(heldout_metadata),
            "data_manifest_sha256": "1" * 64,
            "split_sha256": "2" * 64,
            "selection": "all_validation_examples",
            "entry_count": 1,
            "target_steps": LOCKED_TARGETS["pusht"],
            "rounding_rule": "ceil(target_steps*percent/100)",
        }
    }
    monkeypatch.setattr(
        harness_common, "require_real_marvin_path", lambda value, _label: value
    )
    card = make_manifests.materialize_training_card(
        configured,
        loaded,
        evidence,
        winner="mapanything_recovered_framewise",
        heldout=heldout,
        rates=timing_path,
        producer_decision=producer_decision,
        arm="dinocular",
        environment="pusht",
        seed=1,
    )
    matrix_path = tmp_path / "accepted-matrix.yaml"
    harness_common.write_matrix(
        matrix_path,
        kind="p3-training",
        cards=[card],
        source_commit=evidence["source_commit"],
    )
    matrix, cards = harness_common.load_matrix(matrix_path)
    assert matrix["card_count"] == 1
    card = cards[0]
    run_card_path = Path(matrix["cards"][0]["path"])
    run_dir = Path(card["run_dir"])
    run_dir.mkdir(parents=True)

    def git_output(command, **_kwargs):
        return "f" * 40 + "\n" if "rev-parse" in command else ""

    monkeypatch.setattr(submit_matrix.subprocess, "check_output", git_output)
    monkeypatch.setattr(submit_p3_chain.subprocess, "check_output", git_output)
    monkeypatch.setattr(
        harness_common, "require_real_marvin_path", lambda value, _label: value
    )
    monkeypatch.setattr(
        submit_matrix, "require_real_marvin_path", lambda value, _label: value
    )
    for key, value in card["environment_variables"].items():
        monkeypatch.setenv(key, value)
    submit_matrix.verify_live_card(card)

    chain_manifest = {
        "schema": submit_p3_chain.SCHEMA,
        "run_dir": str(run_dir),
        "run_card": str(run_card_path),
        "run_card_file_sha256": sha256_file(run_card_path),
        "run_card_sha256": card["run_card_sha256"],
        "code_root": str(code_root),
        "source_commit": card["source_commit"],
        "source_file_sha256": card["source_file_sha256"],
        "artifacts": card["artifacts"],
        "container": card["container"],
        "environment_variables": card["environment_variables"],
    }
    chain_path = run_dir / "chain.json"
    _write_json(chain_path, chain_manifest)
    submit_p3_chain.verify(SimpleNamespace(manifest=chain_path))

    reader = EmpiricalDepthCacheReader(
        environment="pusht",
        source_root=source_root,
        empirical_contract_path=depth_inputs["empirical_contract_path"],
        empirical_contract_sha256=depth_inputs["empirical_contract_sha256"],
        empirical_runtime_release_path=depth_inputs[
            "empirical_runtime_release_path"
        ],
        empirical_runtime_release_sha256=depth_inputs[
            "empirical_runtime_release_sha256"
        ],
        expected_checkpoint_sha256=identities["checkpoint_sha256"],
    )
    reader.assert_dataset_coverage([("valid", 0, 1)])
    wire_depth, wire_mask = reader.read(split="valid", episode=0, frames=[0])
    encoder = DinocularEncoder(
        backend="df2_dino_rope_convs_de",
        factory="DFormerv2_S",
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=identities["checkpoint_sha256"],
        checkpoint_key="student",
        state_prefix="module.backbone.",
        input_size=224,
        num_patches=49,
        emb_dim=512,
        empirical_depth_contract_path=depth_inputs["empirical_contract_path"],
        empirical_depth_contract_sha256=depth_inputs["empirical_contract_sha256"],
        empirical_runtime_release_path=depth_inputs[
            "empirical_runtime_release_path"
        ],
        empirical_runtime_release_sha256=depth_inputs[
            "empirical_runtime_release_sha256"
        ],
        depth_input_mode="empirical_lossy_cache_v1",
    )
    features = encoder(
        torch.zeros(1, 3, 224, 224), wire_depth.unsqueeze(0), wire_mask.unsqueeze(0)
    )
    mode, provenance = harness_common.run_card_receipt_expectations(card)
    assert features.shape == (1, 49, 512)
    assert mode == "proxy_depth_z"
    assert provenance == card["depth_inputs"]

def test_valid_authorized_spec_constructs_card_accepted_by_every_authority_surface(
    tmp_path: Path,
) -> None:
    spec, bindings, authorization_sha = _authorization_chain(tmp_path)
    configured = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "conf/study_matrix.yaml").read_text(
            encoding="utf-8"
        )
    )
    configured.update(spec)
    configured["launch_authorization"] = {
        "status": "AUTHORIZED",
        "path": str(tmp_path / "launch-authorization.json"),
        "sha256": authorization_sha,
    }
    configured["authorization_bindings"] = bindings
    evidence = {
        "source_commit": bindings["source_commit"],
        "source_file_sha256": {"train.py": "1" * 64},
        "artifacts": configured["artifacts"],
        "container": configured["container"],
    }
    card = make_manifests._base_card(
        configured,
        evidence,
        kind="p3-training",
        run_id="authorized-card",
        environment="pusht",
        arm="dino_pinned",
        seed=1,
    )
    assert card["launch_authorization_subject"] == configured[
        "launch_authorization_subject"
    ]
    for operation in sorted(harness_common.LAUNCH_OPERATIONS):
        harness_common.require_launch_authorization(card, operation=operation)


@pytest.mark.parametrize(
    ("name", "invalid_state"),
    [
        ("source_release", "PASS"),
        ("empirical_implementation_acceptance", "ACCEPTED"),
        ("immutable_execution_acceptance", "ACCEPTED"),
        ("immutable_probe_acceptance", "PASS"),
    ],
)
def test_authorization_prerequisites_reject_nonexact_state_and_unknown_fields(
    tmp_path: Path, name: str, invalid_state: str
) -> None:
    spec, _bindings, _authorization_sha = _authorization_chain(tmp_path)
    reference = spec["authorization_prerequisites"][name]
    path = Path(reference["path"])
    record = json.loads(path.read_text(encoding="utf-8"))
    record["state"] = invalid_state
    record["unreviewed"] = True
    reference["sha256"] = _write_json(path, record)
    with pytest.raises(harness_common.HarnessError, match="prerequisite"):
        harness_common.resolve_authorization_prerequisites(
            spec,
            contracts_index_sha256="2" * 64,
            empirical_contract_sha256="3" * 64,
            empirical_runtime_release_sha256="4" * 64,
            source_commit="f" * 40,
            source_file_sha256={"train.py": "1" * 64},
        )


@pytest.mark.parametrize(
    "name",
    [
        "empirical_implementation_acceptance",
        "immutable_execution_acceptance",
        "immutable_probe_acceptance",
    ],
)
def test_authorization_acceptances_require_exact_pass_verdict(
    tmp_path: Path, name: str
) -> None:
    spec, _bindings, _authorization_sha = _authorization_chain(tmp_path)
    reference = spec["authorization_prerequisites"][name]
    path = Path(reference["path"])
    record = json.loads(path.read_text(encoding="utf-8"))
    record["verdict"] = "REVIEWED"
    reference["sha256"] = _write_json(path, record)
    with pytest.raises(harness_common.HarnessError, match="prerequisite"):
        harness_common.resolve_authorization_prerequisites(
            spec,
            contracts_index_sha256="2" * 64,
            empirical_contract_sha256="3" * 64,
            empirical_runtime_release_sha256="4" * 64,
            source_commit="f" * 40,
            source_file_sha256={"train.py": "1" * 64},
        )


def test_empirical_contract_rejects_symlink_alias(tmp_path: Path) -> None:
    target = tmp_path / "contract.json"
    digest = _write_json(target, _contract_value(tmp_path))
    alias = tmp_path / "contract-alias.json"
    alias.symlink_to(target)
    with pytest.raises(EmpiricalDepthContractError, match="symlink"):
        load_empirical_depth_contract(alias, digest)


def test_live_verification_rejects_symlinked_host_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code_root = tmp_path / "code"
    code_root.mkdir()
    source = code_root / "train.py"
    source.write_text("pass\n", encoding="utf-8")
    source_alias = code_root / "train-alias.py"
    source_alias.symlink_to(source)
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"artifact")
    card = {
        "run_id": "alias-card",
        "code_root": str(code_root),
        "source_commit": "f" * 40,
        "source_file_sha256": {"train-alias.py": sha256_file(source)},
        "artifacts": {
            "artifact": {"path": str(artifact), "sha256": sha256_file(artifact)}
        },
        "container": {"path": str(artifact), "sha256": sha256_file(artifact)},
    }
    monkeypatch.setattr(
        submit_matrix.subprocess,
        "check_output",
        lambda command, **_kwargs: "f" * 40 + "\n" if "rev-parse" in command else "",
    )
    monkeypatch.setattr(
        submit_matrix, "require_real_marvin_path", lambda value, _label: value
    )
    with pytest.raises(harness_common.HarnessError, match="symlink"):
        submit_matrix.verify_live_card(card)


def test_capsule_runtime_requires_accepted_capsule_and_deployment_evidence(
    tmp_path: Path,
) -> None:
    contract_path = tmp_path / "contract.json"
    contract_sha = _write_json(contract_path, _contract_value(tmp_path))
    contract = load_empirical_depth_contract(contract_path, contract_sha)
    identities = {
        "checkpoint_sha256": contract.checkpoint_sha256,
        "producer_sha256": contract.producer_sha256,
        "manifest_sha256": contract.manifest_sha256,
        "manifest_id": contract.manifest_id,
        "validation_sha256": contract.validation_sha256,
        "data_sha256": contract.data_sha256,
        "source_index_sha256": contract.source_index_sha256,
        "wire_format_sha256": contract.wire_format_sha256,
    }
    canonical_paths = {
        "checkpoint": contract.canonical_paths.checkpoint,
        "cache_directory": contract.canonical_paths.cache_directory,
        "manifest": contract.canonical_paths.manifest,
        "validation": contract.canonical_paths.validation,
        "data": contract.canonical_paths.data,
    }
    runtime_paths = dict(CAPSULE_RUNTIME_PATHS)
    acceptance_path = tmp_path / "runtime-acceptance.json"
    acceptance_sha = _write_json(
        acceptance_path,
        {
            "schema": "dino-wm-empirical-runtime-release-acceptance-v1",
            "state": "INDEPENDENTLY_ACCEPTED",
            "bindings": {
                "empirical_contract_sha256": contract_sha,
                "identities": identities,
                "canonical_paths": canonical_paths,
                "runtime_mode": "capsule_v1",
            },
        },
    )
    release = {
        "schema": "dino-wm-empirical-runtime-release-v1",
        "state": "INDEPENDENTLY_ACCEPTED",
        "empirical_contract_sha256": contract_sha,
        "identities": identities,
        "canonical_paths": canonical_paths,
        "runtime_mode": "capsule_v1",
        "independent_acceptance": {
            "path": str(acceptance_path),
            "sha256": acceptance_sha,
        },
    }
    release_path = tmp_path / "release.json"
    missing_evidence_sha = _write_json(release_path, release)
    with pytest.raises(EmpiricalDepthContractError, match="capsule evidence"):
        load_empirical_runtime_release(
            release_path, missing_evidence_sha, contract=contract
        )
    capsule_path = tmp_path / "capsule.json"
    capsule_sha = _write_json(
        capsule_path,
        {
            "schema": "dino-wm-empirical-capsule-v1",
            "state": "ACCEPTED",
            "empirical_contract_sha256": contract_sha,
            "identities": identities,
            "runtime_paths": runtime_paths,
        },
    )
    deployment_path = tmp_path / "deployment.json"
    deployment_sha = _write_json(
        deployment_path,
        {
            "schema": "dino-wm-empirical-capsule-deployment-acceptance-v1",
            "state": "INDEPENDENTLY_ACCEPTED",
            "verdict": "PASS",
            "bindings": {
                "empirical_contract_sha256": contract_sha,
                "capsule_record_sha256": capsule_sha,
                "runtime_paths": runtime_paths,
            },
        },
    )
    release["capsule"] = {
        "status": "READY",
        "path": str(capsule_path),
        "sha256": capsule_sha,
    }
    release["deployment_acceptance"] = {
        "status": "READY",
        "path": str(deployment_path),
        "sha256": deployment_sha,
    }
    release_sha = _write_json(release_path, release)
    loaded = load_empirical_runtime_release(release_path, release_sha, contract=contract)
    assert loaded.mode == "capsule_v1"
    assert {key: str(value) for key, value in loaded.runtime_paths.items()} == runtime_paths


def test_capsule_paths_flow_through_card_environment_and_live_artifacts(
    tmp_path: Path,
) -> None:
    loaded = _loaded_empirical_index(tmp_path, runtime_mode="capsule_v1")
    inputs = harness_common.depth_inputs(
        loaded, "mapanything_recovered_framewise", "pusht"
    )
    card = {
        "arm": "dinocular",
        "artifacts": {"dinocular_student": {"path": "/canonical/student.pth"}},
        "environment_variables": {},
    }
    make_manifests._depth_overrides(card, inputs)
    assert card["environment_variables"]["DINOCULAR_STUDENT_WEIGHTS"] == (
        CAPSULE_RUNTIME_PATHS["checkpoint"]
    )
    assert card["environment_variables"]["DINOCULAR_EMPIRICAL_DEPTH_CONTRACT"] == (
        CAPSULE_RUNTIME_PATHS["empirical_contract"]
    )
    records = harness_common.depth_artifact_records(card["depth_inputs"])
    assert records["depth cache manifest"]["path"] == str(
        Path(str(inputs["cache_dir"])) / "manifest.json"
    )
    assert records["empirical checkpoint"]["path"] == inputs["checkpoint_path"]
    assert not str(records["depth cache manifest"]["path"]).startswith("/opt/")
    assert not str(records["empirical checkpoint"]["path"]).startswith("/opt/")
    assert records["empirical capsule record"]["path"] == inputs[
        "capsule_record_path"
    ]
    assert records["empirical deployment acceptance"]["path"] == inputs[
        "deployment_acceptance_path"
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        {"depth_inputs": {"contract_kind": None}},
        {"depth_inputs": {"runtime_paths": {"checkpoint": "/opt/forged"}}},
        {"depth_inputs": {"adapter_mode": "proxy_depth_z"}},
        {"depth_inputs": {"empirical_contract_path": "/opt/contract.json"}},
        {"depth_inputs": {"capsule_record_sha256": "a" * 64}},
        {"depth_inputs": {"deployment_acceptance_sha256": "b" * 64}},
        {"environment_variables": {"DINOCULAR_DEPTH_INPUT_MODE": "empirical_lossy_cache_v1"}},
        {"environment_variables": {"DINOCULAR_EMPIRICAL_ADAPTER_MODE": "proxy_depth_z"}},
        {"overrides": ["+env.dataset.empirical_depth_contract_path=/opt/contract.json"]},
        {"overrides": ["+env.dataset.depth_contract_kind=empirical_lossy_cache"]},
        {"empirical_runtime_release": "/opt/release.json"},
    ],
)
def test_legacy_schema_rejects_every_empirical_smuggling_family(mutation: dict) -> None:
    card = {
        "schema": harness_common.LEGACY_RUN_CARD_SCHEMA,
        "arm": "dinocular",
        "environment": "pusht",
        "depth_inputs": {
            "producer": "mapanything_recovered_framewise",
            "producer_sha256": "1" * 64,
            "cache_dir": "/native/pusht.lmdb",
            "cache_manifest_sha256": "2" * 64,
            "validation_path": "/native/validation.json",
            "validation_sha256": "3" * 64,
            "native_contract_path": "/native/contract.json",
            "native_contract_sha256": "4" * 64,
            "checkpoint_sha256": "5" * 64,
        },
        "environment_variables": {
            "DINOV2_REPO": "/native/dinov2",
            "DINOV2_VITS14_WEIGHTS": "/native/dinov2.pth",
            "DINOCULAR_STUDENT_WEIGHTS": "/native/student.pth",
            "DINOCULAR_NATIVE_DEPTH_CONTRACT": "/native/contract.json",
            "DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256": "4" * 64,
            "DINOCULAR_CACHE_PRODUCER_SHA256": "1" * 64,
        },
        "overrides": ["encoder=dinocular"],
    }
    for key, value in mutation.items():
        if isinstance(value, dict) and isinstance(card.get(key), dict):
            card[key] = {**card[key], **value}
        else:
            card[key] = value
    with pytest.raises(harness_common.HarnessError, match="legacy"):
        harness_common.require_run_card_authorization(card, operation="submit")


def test_candidate_classification_requires_nonnull_schema_authority_and_empirical_identity() -> None:
    card = {
        "schema": harness_common.RUN_CARD_SCHEMA,
        "arm": "dinocular",
        "environment": "pusht",
        "launch_authorization_subject": None,
        "launch_authorization": None,
        "authorization_bindings": {},
        "depth_inputs": {"contract_kind": None},
        "environment_variables": {},
        "overrides": ["encoder=dinocular_pusht_empirical"],
    }
    with pytest.raises(harness_common.HarnessError, match="candidate|schema|identity"):
        harness_common.classify_run_card(card)


def test_authority_index_and_run_card_symlinks_are_rejected_before_open(
    tmp_path: Path,
) -> None:
    spec, bindings, authorization_sha = _authorization_chain(tmp_path)
    authorization = tmp_path / "launch-authorization.json"
    authorization_alias = tmp_path / "launch-authorization-alias.json"
    authorization_alias.symlink_to(authorization)
    candidate = {
        **spec,
        "launch_authorization": {
            "status": "AUTHORIZED",
            "path": str(authorization_alias),
            "sha256": authorization_sha,
        },
        "authorization_bindings": bindings,
    }
    with pytest.raises(harness_common.HarnessError, match="symlink"):
        harness_common.require_launch_authorization(candidate, operation="submit")

    contract_path = tmp_path / "contract.json"
    contract_sha = _write_json(contract_path, _contract_value(tmp_path))
    index = {
        "schema": "dino-wm-depth-consumption-index-v2",
        "defaults": None,
        "native_v1": {
            "status": "BLOCKED_MISSING_ACCEPTED_ARTIFACT",
            "index_path": "/missing/native.yaml",
            "index_sha256": None,
            "delegated_environments": ["wall", "rope", "granular"],
        },
        "empirical_runtime_release": {
            "status": "BLOCKED_MISSING_ACCEPTED_ARTIFACT",
            "release_path": "/missing/release.json",
            "release_sha256": None,
        },
        "entries": {
            "pusht/mapanything_recovered_framewise": {
                "contract_kind": "empirical_lossy_cache",
                "contract_path": str(contract_path),
                "contract_sha256": contract_sha,
                "environment": "pusht",
                "allowed_arms": ["dinocular", "dinocular_zerodepth"],
                "adapter_id": EMPIRICAL_ADAPTER_ID,
                "checkpoint_path": _contract_value(tmp_path)["checkpoint"]["path"],
                "checkpoint_sha256": CHECKPOINT_SHA,
                "cache_dir": _contract_value(tmp_path)["cache"]["directory"],
                "producer_sha256": PRODUCER_SHA,
                "manifest_sha256": MANIFEST_SHA,
                "manifest_id": MANIFEST_ID,
                "validation_path": _contract_value(tmp_path)["cache"]["validation"]["path"],
                "validation_sha256": RECEIPT_SHA,
                "validation_schema": "dinocular-mapanything-cache-validation-v1",
                "data_path": _contract_value(tmp_path)["cache"]["data_path"],
                "data_sha256": DATA_SHA,
                "source_index_sha256": SOURCE_SHA,
                "wire_format_sha256": WIRE_SHA,
                "assumption_tags": [EMPIRICAL_ASSUMPTION],
                "execution_authority_granted": False,
            }
        },
    }
    index_path = tmp_path / "index.yaml"
    index_path.write_text(yaml.safe_dump(index, sort_keys=False), encoding="utf-8")
    index_alias = tmp_path / "index-alias.yaml"
    index_alias.symlink_to(index_path)
    with pytest.raises(EmpiricalDepthContractError, match="symlink"):
        load_depth_consumption_index(index_alias)

    run_card_path = tmp_path / "run-card.yaml"
    run_card_path.write_text("schema: forged\n", encoding="utf-8")
    run_card_alias = tmp_path / "run-card-alias.yaml"
    run_card_alias.symlink_to(run_card_path)
    with pytest.raises((RuntimeError, harness_common.HarnessError), match="symlink"):
        submit_p3_chain.load_and_validate_run_card(
            run_card_alias,
            expected_file_sha256=sha256_file(run_card_path),
            expected_content_sha256="a" * 64,
        )


def test_chain_manifest_and_checkpoint_aliases_are_rejected_before_open(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "chain.json"
    _write_json(
        manifest_path,
        {"schema": submit_p3_chain.SCHEMA, "run_dir": str(tmp_path)},
    )
    manifest_alias = tmp_path / "chain-alias.json"
    manifest_alias.symlink_to(manifest_path)
    with pytest.raises(RuntimeError, match="symlink"):
        submit_p3_chain.load_manifest(manifest_alias)

    checkpoint_dir = tmp_path / "checkpoints" / "steps"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_target = checkpoint_dir / "checkpoint-target.pth"
    checkpoint_target.write_bytes(b"checkpoint")
    checkpoint_alias = checkpoint_dir / "step_000000001.pth"
    checkpoint_alias.symlink_to(checkpoint_target)
    progress = {
        "immutable_run_card_sha256": "a" * 64,
        "training_process_id": 123,
        "global_step": 1,
        "checkpoint": str(checkpoint_alias),
        "checkpoint_sha256": sha256_file(checkpoint_target),
    }
    with pytest.raises(RuntimeError, match="symlink"):
        submit_p3_chain.verify_progress_evidence(
            {
                "run_card_sha256": "a" * 64,
                "run_dir": str(tmp_path),
            },
            progress,
        )


def test_receipt_expectations_come_from_validated_schema_arm_and_encoder(
    tmp_path: Path,
) -> None:
    configured = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "conf/study_matrix.yaml").read_text(
            encoding="utf-8"
        )
    )
    configured["launch_authorization_subject"] = "pusht-empirical-candidate-v1"
    configured["launch_authorization"] = {
        "status": "AUTHORIZED",
        "path": str(tmp_path / "authorization.json"),
        "sha256": "a" * 64,
    }
    configured["authorization_bindings"] = {
        name: ("f" * 40 if name == "source_commit" else "b" * 64)
        for name in harness_common.AUTHORIZATION_BINDING_FIELDS
    }
    card = make_manifests._base_card(
        configured,
        {
            "source_commit": "f" * 40,
            "source_file_sha256": {},
            "artifacts": configured["artifacts"],
            "container": configured["container"],
        },
        kind="p3-training",
        run_id="receipt-schema",
        environment="pusht",
        arm="dinocular",
        seed=1,
    )
    inputs = harness_common.depth_inputs(
        _loaded_empirical_index(tmp_path),
        "mapanything_recovered_framewise",
        "pusht",
    )
    make_manifests._depth_overrides(card, inputs)
    card["overrides"] = ["encoder=dinocular_pusht_empirical"]
    mode, provenance = harness_common.run_card_receipt_expectations(card)
    assert mode == "proxy_depth_z"
    assert provenance == card["depth_inputs"]

    smuggled = copy.deepcopy(card)
    smuggled["depth_inputs"]["contract_kind"] = None
    with pytest.raises(harness_common.HarnessError, match="schema|identity"):
        harness_common.run_card_receipt_expectations(smuggled)


def test_loader_output_rejects_document_and_runtime_binding_mutation(
    tmp_path: Path,
) -> None:
    loaded = _loaded_empirical_index(tmp_path, runtime_mode="capsule_v1")
    entry_name = "pusht/mapanything_recovered_framewise"
    before = harness_common.depth_inputs(
        loaded, "mapanything_recovered_framewise", "pusht"
    )
    with pytest.raises(TypeError):
        loaded["entries"][entry_name]["cache_dir"] = "/forged/cache.lmdb"
    binding = loaded.runtime_binding(entry_name)
    with pytest.raises(TypeError):
        binding.runtime_paths["checkpoint"] = "/opt/forged.pth"
    after = harness_common.depth_inputs(
        loaded, "mapanything_recovered_framewise", "pusht"
    )
    assert after == before


def test_capsule_host_artifacts_are_canonical_and_never_opt_paths() -> None:
    value = {
        **_provenance("a" * 64),
        "empirical_contract_path": "/canonical/contract.json",
        "empirical_runtime_release_path": "/canonical/release.json",
        "empirical_runtime_release_sha256": "b" * 64,
        "runtime_mode": "capsule_v1",
        "runtime_paths": dict(CAPSULE_RUNTIME_PATHS),
        "cache_dir": "/canonical/pusht.lmdb",
        "validation_path": "/canonical/validation.json",
        "checkpoint_path": "/canonical/student.pth",
        "capsule_record_path": "/canonical/capsule.json",
        "capsule_record_sha256": "c" * 64,
        "deployment_acceptance_path": "/canonical/deployment.json",
        "deployment_acceptance_sha256": "d" * 64,
    }
    records = harness_common.depth_artifact_records(value)
    assert all(not str(record["path"]).startswith("/opt/") for record in records.values())
    assert records["depth cache manifest"]["path"] == "/canonical/pusht.lmdb/manifest.json"
    assert records["empirical checkpoint"]["path"] == "/canonical/student.pth"
    assert records["empirical capsule record"]["path"] == "/canonical/capsule.json"
    assert records["empirical deployment acceptance"]["path"] == "/canonical/deployment.json"
