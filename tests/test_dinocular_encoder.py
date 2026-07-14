from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from models import dinocular_backbone
from models.dinocular import (
    CheckpointHashError,
    CheckpointLoadError,
    DinocularEncoder,
    load_backbone_checkpoint,
    sha256_file,
)
from models.dinocular_backbone import BackendSpec, build_backbone


class TinyExactBackbone(nn.Module):
    def __init__(self, **_: object) -> None:
        super().__init__()
        self.rgb = nn.Conv2d(3, 8, kernel_size=1)
        self.depth = nn.Conv2d(1, 8, kernel_size=1)
        self.bn = nn.BatchNorm2d(8)

    def forward_features(self, rgb: torch.Tensor, depth: torch.Tensor):
        feature_map = self.bn(self.rgb(rgb) + self.depth(depth))
        feature_map = F.adaptive_avg_pool2d(feature_map, (1, 1))
        tokens = feature_map.flatten(2).transpose(1, 2)
        return {
            "x_norm_patchtokens": tokens,
            "x_norm_clstoken": tokens.mean(dim=1),
        }


@pytest.fixture
def tiny_backend(monkeypatch: pytest.MonkeyPatch) -> BackendSpec:
    spec = BackendSpec(
        module=SimpleNamespace(TinyExact=TinyExactBackbone),
        factories={"TinyExact": 8},
        output_kind="dino_feature_dict",
        checkpoint_policy="exact",
    )
    monkeypatch.setitem(dinocular_backbone.BACKEND_REGISTRY, "tiny_exact", spec)
    return spec


def _save_checkpoint(path: Path, value: object) -> str:
    torch.save(value, path)
    return sha256_file(path)


def _make_tiny_encoder(tmp_path: Path, tiny_backend: BackendSpec) -> DinocularEncoder:
    source = TinyExactBackbone()
    checkpoint_path = tmp_path / "tiny.pth"
    state = {f"module.backbone.{key}": tensor.clone() for key, tensor in source.state_dict().items()}
    state["module.head.weight"] = torch.ones(2, 2)
    checksum = _save_checkpoint(checkpoint_path, {"teacher": state})
    return DinocularEncoder(
        backend="tiny_exact",
        factory="TinyExact",
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=checksum,
        checkpoint_key="teacher",
        state_prefix="module.backbone.",
        feature_key="x_norm_patchtokens",
        input_size=32,
        num_patches=1,
        emb_dim=8,
        frozen=True,
        depth_contract={"temporal_mode": "complete_trajectory"},
        allowed_outside_prefixes=["module.head."],
    )


def test_vendored_sources_match_pinned_mthesis_files() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        "_vendor_dformerv2_stock.py": "aaeb1992312ebc9738279edd984f8e6f0b238f6e3f6d344074256f515ec17b77",
        "_vendor_df2_dino_rope_convs_de.py": "2dab217777185d19b144be22fafe80dc941fd0a67ae9a6e5fc5cd380b48ab1e6",
    }
    for filename, checksum in expected.items():
        assert sha256_file(root / "models" / filename) == checksum


def test_registry_is_closed_and_real_factories_disable_drop_path() -> None:
    with pytest.raises(ValueError, match="Unknown Dinocular backend"):
        build_backbone("not_registered", "DFormerv2_S")
    with pytest.raises(ValueError, match="is not registered"):
        build_backbone("dformerv2_stock", "made_up_factory")

    backbone, _ = build_backbone("dformerv2_stock", "DFormerv2_S")
    assert all(
        float(module.drop_prob) == 0.0
        for module in backbone.modules()
        if hasattr(module, "drop_prob")
    )


def test_exact_load_native_tokens_normalization_and_frozen_eval(
    tmp_path: Path, tiny_backend: BackendSpec
) -> None:
    encoder = _make_tiny_encoder(tmp_path, tiny_backend)
    assert encoder.input_size == 32
    assert encoder.num_patches == 1
    assert encoder.emb_dim == 8
    assert encoder.latent_ndim == 2
    assert encoder.load_audit.allowed_outside_prefix == ("module.head.weight",)
    assert all(not parameter.requires_grad for parameter in encoder.parameters())

    rgb = torch.zeros(2, 3, 32, 32)
    depth = torch.full((2, 32, 32), 0.48)
    rgb_prepared = encoder.preprocess_rgb(rgb)
    expected_rgb = (
        torch.full_like(rgb, 0.5)
        - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    ) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    torch.testing.assert_close(rgb_prepared, expected_rgb)
    torch.testing.assert_close(encoder.preprocess_depth(depth), torch.zeros(2, 1, 32, 32))

    before_mean = encoder.backbone.bn.running_mean.clone()
    output_a = encoder(rgb, depth=depth)
    output_b = encoder(rgb, depth=depth)
    assert output_a.shape == (2, 1, 8)
    torch.testing.assert_close(output_a, output_b, rtol=0, atol=0)
    torch.testing.assert_close(encoder.backbone.bn.running_mean, before_mean, rtol=0, atol=0)

    returned = encoder.train(True)
    assert returned is encoder
    assert not encoder.training
    assert not encoder.backbone.training
    assert not encoder.backbone.bn.training


def test_depth_is_required_and_inputs_fail_closed(
    tmp_path: Path, tiny_backend: BackendSpec
) -> None:
    encoder = _make_tiny_encoder(tmp_path, tiny_backend)
    rgb = torch.zeros(1, 3, 32, 32)
    depth = torch.zeros(1, 1, 32, 32)
    with pytest.raises(ValueError, match="isolated-frame/zero-depth fallback is forbidden"):
        encoder(rgb)
    with pytest.raises(ValueError, match="outside declared"):
        encoder(rgb + 2.0, depth=depth)
    with pytest.raises(ValueError, match="outside declared"):
        encoder(rgb, depth=depth - 0.1)
    with pytest.raises(ValueError, match="spatial size"):
        encoder(rgb[:, :, :-1], depth=depth[:, :, :-1])


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "shape"])
def test_exact_checkpoint_rejects_every_backbone_mismatch(
    tmp_path: Path, tiny_backend: BackendSpec, mutation: str
) -> None:
    source = TinyExactBackbone()
    state = {f"module.backbone.{key}": tensor.clone() for key, tensor in source.state_dict().items()}
    if mutation == "missing":
        del state["module.backbone.rgb.weight"]
    elif mutation == "unexpected":
        state["module.backbone.not_a_parameter"] = torch.ones(1)
    else:
        state["module.backbone.rgb.weight"] = torch.ones(9, 3, 1, 1)
    path = tmp_path / f"{mutation}.pth"
    checksum = _save_checkpoint(path, {"teacher": state})
    with pytest.raises(CheckpointLoadError):
        DinocularEncoder(
            backend="tiny_exact",
            factory="TinyExact",
            checkpoint_path=str(path),
            checkpoint_sha256=checksum,
            checkpoint_key="teacher",
            state_prefix="module.backbone.",
            input_size=32,
            num_patches=1,
            emb_dim=8,
        )


def test_exact_checkpoint_rejects_undeclared_wrapper_tensors(
    tmp_path: Path, tiny_backend: BackendSpec
) -> None:
    source = TinyExactBackbone()
    state = {
        f"module.backbone.{key}": tensor.clone()
        for key, tensor in source.state_dict().items()
    }
    state["module.head.weight"] = torch.ones(2, 2)
    path = tmp_path / "wrapper.pth"
    checksum = _save_checkpoint(path, {"teacher": state})
    with pytest.raises(CheckpointLoadError, match="outside the declared backbone prefix"):
        DinocularEncoder(
            backend="tiny_exact",
            factory="TinyExact",
            checkpoint_path=str(path),
            checkpoint_sha256=checksum,
            checkpoint_key="teacher",
            state_prefix="module.backbone.",
            input_size=32,
            num_patches=1,
            emb_dim=8,
        )


def test_hash_mismatch_stops_before_deserialization(
    tmp_path: Path, tiny_backend: BackendSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "artifact.pth"
    path.write_bytes(b"not a torch checkpoint")

    def forbidden_load(*_: object, **__: object) -> object:
        raise AssertionError("torch.load must not run after a hash mismatch")

    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(CheckpointHashError, match="SHA-256 mismatch"):
        DinocularEncoder(
            backend="tiny_exact",
            factory="TinyExact",
            checkpoint_path=str(path),
            checkpoint_sha256="0" * 64,
            checkpoint_key="teacher",
            state_prefix="module.backbone.",
            input_size=32,
            num_patches=1,
            emb_dim=8,
        )


class _Pred(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Sequential(nn.Conv2d(4, 4, kernel_size=1))


class StockAuditBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, 4))
        self.used = nn.Linear(4, 4)
        self.extra_norms = nn.LayerNorm(4)
        self.pred = _Pred()


def _stock_checkpoint_state(backbone: StockAuditBackbone) -> dict[str, torch.Tensor]:
    state = {}
    for key, tensor in backbone.state_dict().items():
        if key in {"mask_token", "pred.decoder.0.weight", "pred.decoder.0.bias"}:
            continue
        if key.startswith("extra_norms."):
            key = "extra_norms.2." + key.removeprefix("extra_norms.")
        state[f"backbone.{key}"] = tensor.clone()
    state["backbone.extra_norms.0.weight"] = torch.ones(2)
    state["backbone.extra_norms.0.bias"] = torch.zeros(2)
    state["backbone.extra_norms.1.weight"] = torch.ones(3)
    state["backbone.extra_norms.1.bias"] = torch.zeros(3)
    state["decode_head.weight"] = torch.ones(1)
    return state


def test_stock_public_mapping_has_only_the_documented_allowlist(tmp_path: Path) -> None:
    backbone = StockAuditBackbone()
    state = _stock_checkpoint_state(backbone)
    path = tmp_path / "stock.pth"
    checksum = _save_checkpoint(path, {"state_dict": state})
    spec = BackendSpec(
        module=SimpleNamespace(),
        factories={"unused": 4},
        output_kind="stock_feature_map",
        checkpoint_policy="stock_sunrgbd",
    )
    audit = load_backbone_checkpoint(
        backbone,
        spec,
        path,
        checksum,
        checkpoint_key="state_dict",
        state_prefix="backbone.",
    )
    assert audit.allowed_model_only == (
        "mask_token",
        "pred.decoder.0.bias",
        "pred.decoder.0.weight",
    )
    assert audit.allowed_checkpoint_only == (
        "extra_norms.0.bias",
        "extra_norms.0.weight",
        "extra_norms.1.bias",
        "extra_norms.1.weight",
    )
    assert audit.allowed_outside_prefix == ("decode_head.weight",)

    state["backbone.silent_partial_load"] = torch.ones(1)
    bad_path = tmp_path / "stock_bad.pth"
    bad_checksum = _save_checkpoint(bad_path, {"state_dict": state})
    with pytest.raises(CheckpointLoadError, match="unexpected"):
        load_backbone_checkpoint(
            StockAuditBackbone(),
            spec,
            bad_path,
            bad_checksum,
            checkpoint_key="state_dict",
            state_prefix="backbone.",
        )

    outside_state = _stock_checkpoint_state(StockAuditBackbone())
    outside_state["auxiliary_head.weight"] = torch.ones(1)
    outside_path = tmp_path / "stock_outside_bad.pth"
    outside_checksum = _save_checkpoint(outside_path, {"state_dict": outside_state})
    with pytest.raises(CheckpointLoadError, match="outside the declared backbone prefix"):
        load_backbone_checkpoint(
            StockAuditBackbone(),
            spec,
            outside_path,
            outside_checksum,
            checkpoint_key="state_dict",
            state_prefix="backbone.",
        )


def test_metadata_must_match_native_grid_and_registered_width(
    tmp_path: Path, tiny_backend: BackendSpec
) -> None:
    source = TinyExactBackbone()
    path = tmp_path / "tiny.pth"
    checksum = _save_checkpoint(path, {"teacher": source.state_dict()})
    common = dict(
        backend="tiny_exact",
        factory="TinyExact",
        checkpoint_path=str(path),
        checkpoint_sha256=checksum,
        checkpoint_key="teacher",
        state_prefix="",
        input_size=32,
    )
    with pytest.raises(ValueError, match="num_patches"):
        DinocularEncoder(**common, num_patches=49, emb_dim=8)
    with pytest.raises(ValueError, match="emb_dim"):
        DinocularEncoder(**common, num_patches=1, emb_dim=9)


@pytest.mark.skipif(
    not os.environ.get("DFORMERV2_SUNRGBD_WEIGHTS"),
    reason="set DFORMERV2_SUNRGBD_WEIGHTS for the pinned-artifact audit",
)
def test_pinned_public_artifact_strict_load() -> None:
    path = os.environ["DFORMERV2_SUNRGBD_WEIGHTS"]
    backbone, spec = build_backbone("dformerv2_stock", "DFormerv2_S")
    audit = load_backbone_checkpoint(
        backbone,
        spec,
        path,
        "ba7b95735a3ee032041da44f1ad1290649df5a204a3561e6592f3357cb9cf1ba",
        checkpoint_key="state_dict",
        state_prefix="backbone.",
    )
    assert len(audit.allowed_outside_prefix) == 22
    assert all(key.startswith("decode_head.") for key in audit.allowed_outside_prefix)
    assert audit.allowed_model_only == (
        "mask_token",
        "pred.decoder.0.bias",
        "pred.decoder.0.weight",
    )
