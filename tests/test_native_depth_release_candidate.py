"""Focused loader and materializer tests for the native-v1 depth release candidate.

These tests prove the deterministic native-v1 release candidate for Wall, Rope,
and Granular binds exactly the pinned, independently accepted DA3 depth caches
(and nothing else), that the candidate is not self-accepted, and that the PushT
empirical route is untouched. Every constant below is the read-only Marvin
ground-truth value re-derived at candidate-authoring time; the tests fail if the
candidate artifact drifts away from any pinned identity.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = REPO_ROOT / "contracts" / "native_depth_release_candidate_v1.yaml"
V2_INDEX_PATH = REPO_ROOT / "contracts" / "depth_consumption_index_v2.yaml"

STUDENT_CHECKPOINT_SHA256 = (
    "decc7c73283bf46f66dbbedec6fb065ad0a943fb2ddb1c49317fafb0319f5dcc"
)
# Shared DA3 wire format; identical to the PushT MapAnything empirical wire format
# recorded in contracts/depth_consumption_index_v2.yaml, so all four environments
# speak the same cache wire format.
DA3_WIRE_FORMAT_SHA256 = (
    "47a6d5944af9f3587ee7ef0b8154d157294e91db7e489c06b5b33bd3d9384ded"
)
MARVIN_PREFIX = (
    "/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm"
)

# Read-only Marvin ground truth (2026-07-27): sha256sum of on-disk artifacts plus
# canonical-JSON producer/wire hashes taken from each manifest.
PINNED = {
    "wall": {
        "cache_dir": f"{MARVIN_PREFIX}/data/depth_cache/wall.lmdb",
        "manifest_id": "54bab01a-8c49-4df5-a1d9-3a76e32e5234",
        "manifest_sha256": (
            "b6db2c8b495ed2a3a142973cdc3622ab193a5a38d51147515d2f38678214819d"
        ),
        "data_mdb_sha256": (
            "e4107dc52f50dc206df6bc5c5e8b1655c3b699fa58d6d3196d5329d63c0dc467"
        ),
        "source_index_sha256": (
            "07029278c318ff4aff8fe89fb1b1bd9162e2d93e2635131e0ab4ca9d5285c4a1"
        ),
        "producer_sha256": (
            "4d15ed3f04ea318d72599d1d84653d57ae93ec81deb638272d7b823c7fdf5f54"
        ),
        "tool_sha256": (
            "6d7aa3d5c78592871106efa8123057cba21a8014d4dc9f3a63c16fcb444f44ce"
        ),
        "validation_path": (
            f"{MARVIN_PREFIX}/data/depth_cache_validation/wall.recovery.26548609.json"
        ),
        "validation_sha256": (
            "b2404b80ba54f7b9b9cd857c4032bf515a91d1a3295f62167c7c9c65f122fab4"
        ),
        "calibration_lo": 0.6311277413368225,
        "calibration_hi": 1.145139434337616,
        "calibration_keys_sha256": (
            "ee1c63ffb352cac189111884cbbad331b19ba8ba60abc5de592eaaa77809726d"
        ),
        "trajectory_count": 1920,
        "frame_count": 96000,
    },
    "rope": {
        "cache_dir": f"{MARVIN_PREFIX}/data/depth_cache/rope.lmdb",
        "manifest_id": "0a31897e-f64a-4447-8c7e-0f5db66d9fb3",
        "manifest_sha256": (
            "5eb0818c6d38d6f6c5d210075e1be0ff008d5b04252926986979ddbb6c8f2dba"
        ),
        "data_mdb_sha256": (
            "64cab2bea1b51a053d27bdcd68d3e32d5b65fac15f40e46d62460d6c55af680d"
        ),
        "source_index_sha256": (
            "c8a94086102f5003e7e16934ae28c1ef44bb70d74186bb778b3cb003f8387f1e"
        ),
        "producer_sha256": (
            "c7973c37e7f0c859644627a09f57a02aa64582c84b1a3b59cca2836b02206ecc"
        ),
        "tool_sha256": (
            "5712422eae014e9aeb03fd44b98d33bd2c28f518e9e962d8fb64ac147cb1b79b"
        ),
        "validation_path": (
            f"{MARVIN_PREFIX}/data/depth_cache_validation/"
            "rope.0a31897e-f64a-4447-8c7e-0f5db66d9fb3.json"
        ),
        "validation_sha256": (
            "3df12b04aa0ce2a68484dc023773d3eb7b3bd158ea4a9212c660caeeb5478a86"
        ),
        "calibration_lo": 0.9579539752006531,
        "calibration_hi": 2.320253071784973,
        "calibration_keys_sha256": (
            "2d9088fdefe2277031b6c73573ae36de12903fe1a3dbbce88d10c61000b214a8"
        ),
        "trajectory_count": 1000,
        "frame_count": 20000,
    },
    "granular": {
        "cache_dir": f"{MARVIN_PREFIX}/data/depth_cache/granular.lmdb",
        "manifest_id": "52897463-5cc8-4f70-b999-5f95af9d8b3f",
        "manifest_sha256": (
            "8ba955aa806fc4c1ef8a8ed8a6c282bf1e9d010e99f77ebd4e65d44360e99997"
        ),
        "data_mdb_sha256": (
            "4b6b32385295a2ccd05666730f05f17649e4e70ebce37863d474550558c23349"
        ),
        "source_index_sha256": (
            "51f64e3e7d9f4579f626d804bd445e9f9e6f8744ded757802ea32189a770048d"
        ),
        "producer_sha256": (
            "c7973c37e7f0c859644627a09f57a02aa64582c84b1a3b59cca2836b02206ecc"
        ),
        "tool_sha256": (
            "5712422eae014e9aeb03fd44b98d33bd2c28f518e9e962d8fb64ac147cb1b79b"
        ),
        "validation_path": (
            f"{MARVIN_PREFIX}/data/depth_cache_validation/"
            "granular.52897463-5cc8-4f70-b999-5f95af9d8b3f.json"
        ),
        "validation_sha256": (
            "db67a785b67bedaf3c617b6da74250e0730adbf0f956f1cafd76207451caaec8"
        ),
        "calibration_lo": 0.8734745681285858,
        "calibration_hi": 2.3549521017074584,
        "calibration_keys_sha256": (
            "b2b1273493dc438535aeb35926f8435f05ddda145131447ed272de5cf012bd22"
        ),
        "trajectory_count": 1000,
        "frame_count": 20000,
    },
}

# Report-27 native-upper-bound quarantine cache; a DIFFERENT (PyFlex) producer.
# The DA3 Rope entry must never collide with this path or manifest.
ROPE_NATIVE_UPPER_BOUND_QUARANTINE = (
    f"{MARVIN_PREFIX}/data/depth_cache/native_upper_bound/quarantine/"
    "cpu-native-depth-canary-rope-158c7470-v4/rope.lmdb"
)
ROPE_NATIVE_UPPER_BOUND_MANIFEST_ID = (
    "0e4757f36548fe5836050524bb2ec542fb974497e70c0aee30d4fba9b4fd1e1c"
)


def _load_candidate() -> dict:
    return yaml.safe_load(CANDIDATE_PATH.read_text(encoding="utf-8"))


def test_candidate_schema_marks_it_not_self_accepted() -> None:
    candidate = _load_candidate()
    assert candidate["schema"] == "dinocular-native-depth-release-candidate-v1"
    assert candidate["status"] == "CANDIDATE_NOT_SELF_ACCEPTED"
    assert candidate["promotion_target_schema"] == "dino-wm-depth-contract-index-v1"
    assert candidate["release_kind"] == "native_v1"
    assert candidate["execution_authority_granted"] is False
    assert candidate["self_acceptance_recorded"] is False


def test_candidate_delegates_only_wall_rope_granular() -> None:
    candidate = _load_candidate()
    assert candidate["delegated_environments"] == ["wall", "rope", "granular"]
    assert "pusht" not in candidate["delegated_environments"]
    assert set(candidate["entries"]) == {
        "wall/da3_giant_video",
        "rope/da3_giant_video",
        "granular/da3_giant_video",
    }
    assert candidate["producer"]["name"] == "da3_giant_video"
    assert candidate["producer"]["checkpoint_sha256"] == STUDENT_CHECKPOINT_SHA256


def test_candidate_native_contract_prerequisite_is_honestly_missing() -> None:
    candidate = _load_candidate()
    native_contract = candidate["native_contract"]
    assert native_contract["required_schema"] == "dinocular-native-depth-contract-v1"
    assert native_contract["sha256"] is None
    assert native_contract["state"] == "MISSING_NOT_ACCEPTED"
    # The candidate must not invent a native-contract identity.
    assert native_contract["intended_path"].startswith(MARVIN_PREFIX)


@pytest.mark.parametrize("environment", ["wall", "rope", "granular"])
def test_candidate_entry_binds_exact_pinned_marvin_identity(environment: str) -> None:
    candidate = _load_candidate()
    entry = candidate["entries"][f"{environment}/da3_giant_video"]
    expected = PINNED[environment]
    assert entry["environment"] == environment
    assert entry["contract_kind"] == "native_v1_da3_cache"
    assert entry["producer"] == "da3_giant_video"
    assert entry["cache_dir"] == expected["cache_dir"]
    assert entry["manifest_path"] == f"{expected['cache_dir']}/manifest.json"
    assert entry["manifest_id"] == expected["manifest_id"]
    assert entry["manifest_sha256"] == expected["manifest_sha256"]
    assert entry["data_mdb_path"] == f"{expected['cache_dir']}/data.mdb"
    assert entry["data_mdb_sha256"] == expected["data_mdb_sha256"]
    assert entry["source_index_sha256"] == expected["source_index_sha256"]
    assert entry["producer_sha256"] == expected["producer_sha256"]
    assert entry["wire_format_sha256"] == DA3_WIRE_FORMAT_SHA256
    assert entry["tool_sha256"] == expected["tool_sha256"]
    assert entry["validation_path"] == expected["validation_path"]
    assert entry["validation_sha256"] == expected["validation_sha256"]
    assert entry["validation_schema"] == "dinocular-depth-cache-validation-v1"
    assert entry["calibration"]["lo"] == expected["calibration_lo"]
    assert entry["calibration"]["hi"] == expected["calibration_hi"]
    assert entry["calibration"]["keys_sha256"] == expected["calibration_keys_sha256"]
    assert entry["trajectory_count"] == expected["trajectory_count"]
    assert entry["frame_count"] == expected["frame_count"]


def test_rope_entry_is_da3_not_native_upper_bound_quarantine() -> None:
    candidate = _load_candidate()
    rope = candidate["entries"]["rope/da3_giant_video"]
    # The pinned DA3 Rope cache is manifest 0a31897e at the canonical cache dir.
    assert rope["manifest_id"] == "0a31897e-f64a-4447-8c7e-0f5db66d9fb3"
    assert rope["cache_dir"] == f"{MARVIN_PREFIX}/data/depth_cache/rope.lmdb"
    assert rope["cache_dir"] != ROPE_NATIVE_UPPER_BOUND_QUARANTINE
    assert rope["manifest_id"] != ROPE_NATIVE_UPPER_BOUND_MANIFEST_ID
    # No entry may reference the report-27 quarantine cache.
    for entry in candidate["entries"].values():
        assert ROPE_NATIVE_UPPER_BOUND_QUARANTINE not in entry["cache_dir"]
    # No entry may use staging/quarantine/alias naming for the canonical caches.
    for entry in candidate["entries"].values():
        assert "/quarantine/" not in entry["cache_dir"]
        assert "/." not in entry["cache_dir"]  # no hidden/staging dirs


def test_da3_wire_format_is_identical_to_pusht_mapanything_wire_format() -> None:
    candidate = _load_candidate()
    v2 = yaml.safe_load(V2_INDEX_PATH.read_text(encoding="utf-8"))
    pusht_wire = v2["entries"]["pusht/mapanything_recovered_framewise"][
        "wire_format_sha256"
    ]
    assert pusht_wire == DA3_WIRE_FORMAT_SHA256
    for entry in candidate["entries"].values():
        assert entry["wire_format_sha256"] == pusht_wire


def test_all_entries_are_real_marvin_canonical_cache_paths() -> None:
    candidate = _load_candidate()
    for environment in ("wall", "rope", "granular"):
        entry = candidate["entries"][f"{environment}/da3_giant_video"]
        assert entry["cache_dir"] == (
            f"{MARVIN_PREFIX}/data/depth_cache/{environment}.lmdb"
        )
        assert entry["validation_path"].startswith(
            f"{MARVIN_PREFIX}/data/depth_cache_validation/"
        )


class CandidateNotLoadable(Exception):
    """Raised when the native-v1 candidate cannot resolve an environment."""


def materialize_native_depth_inputs(candidate: dict, environment: str) -> dict:
    """Mirror the native-v1 branch of harness_common.depth_inputs.

    Resolve the exact pinned cache identity for a delegated environment from the
    candidate index. PushT and any non-delegated environment must fail closed,
    exactly as the unchanged native-v1 dispatcher requires.
    """

    if environment not in candidate["delegated_environments"]:
        raise CandidateNotLoadable(
            f"native-v1 route does not delegate {environment!r}"
        )
    entry = candidate["entries"].get(f"{environment}/da3_giant_video")
    if not isinstance(entry, dict):
        raise CandidateNotLoadable(
            f"no pinned DA3 cache entry for {environment!r}"
        )
    return {
        "producer": entry["producer"],
        "producer_sha256": entry["producer_sha256"],
        "cache_dir": entry["cache_dir"],
        "cache_manifest_sha256": entry["manifest_sha256"],
        "validation_path": entry["validation_path"],
        "validation_sha256": entry["validation_sha256"],
        "wire_format_sha256": entry["wire_format_sha256"],
        "native_contract_path": candidate["native_contract"]["intended_path"],
        "native_contract_sha256": candidate["native_contract"]["sha256"],
        "checkpoint_sha256": candidate["producer"]["checkpoint_sha256"],
        "execution_authority_granted": candidate["execution_authority_granted"],
    }


@pytest.mark.parametrize("environment", ["wall", "rope", "granular"])
def test_materializer_resolves_exact_pinned_identity(environment: str) -> None:
    candidate = _load_candidate()
    inputs = materialize_native_depth_inputs(candidate, environment)
    expected = PINNED[environment]
    assert inputs["producer"] == "da3_giant_video"
    assert inputs["producer_sha256"] == expected["producer_sha256"]
    assert inputs["cache_dir"] == expected["cache_dir"]
    assert inputs["cache_manifest_sha256"] == expected["manifest_sha256"]
    assert inputs["validation_path"] == expected["validation_path"]
    assert inputs["validation_sha256"] == expected["validation_sha256"]
    assert inputs["wire_format_sha256"] == DA3_WIRE_FORMAT_SHA256
    assert inputs["checkpoint_sha256"] == STUDENT_CHECKPOINT_SHA256
    # The candidate grants no execution authority along this path.
    assert inputs["execution_authority_granted"] is False
    assert inputs["native_contract_sha256"] is None


def test_materializer_fail_closes_for_pusht_and_unknown_environments() -> None:
    candidate = _load_candidate()
    # PushT is intentionally NOT delegated on the native-v1 route; it has its own
    # empirical route in the v2 consumption index.
    with pytest.raises(CandidateNotLoadable):
        materialize_native_depth_inputs(candidate, "pusht")
    with pytest.raises(CandidateNotLoadable):
        materialize_native_depth_inputs(candidate, "kitchen")


def test_pusht_empirical_route_is_untouched() -> None:
    v2 = yaml.safe_load(V2_INDEX_PATH.read_text(encoding="utf-8"))
    # The canonical dispatcher still marks the native-v1 and empirical runtime
    # releases as blocked with no recorded identity: this candidate does not
    # self-accept and does not edit the PushT empirical path.
    assert v2["native_v1"]["status"] == "BLOCKED_MISSING_ACCEPTED_ARTIFACT"
    assert v2["native_v1"]["index_sha256"] is None
    assert v2["native_v1"]["delegated_environments"] == ["wall", "rope", "granular"]
    assert (
        v2["empirical_runtime_release"]["status"]
        == "BLOCKED_MISSING_ACCEPTED_ARTIFACT"
    )
    pusht = v2["entries"]["pusht/mapanything_recovered_framewise"]
    assert pusht["contract_kind"] == "empirical_lossy_cache"
    assert pusht["environment"] == "pusht"
    assert pusht["execution_authority_granted"] is False
    # The PushT entry set is exactly the one empirical route; no native entries
    # were added to the canonical v2 index by this candidate.
    assert set(v2["entries"]) == {"pusht/mapanything_recovered_framewise"}
