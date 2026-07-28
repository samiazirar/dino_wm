"""Strict RGB-D encoder adapter for stock DFormerv2 and DINOcular backends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Callable, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from depth_contract import DepthContractError, load_native_depth_contract
from empirical_depth_contract import (
    EMPIRICAL_ADAPTER_ID,
    apply_empirical_depth_adapter,
    load_empirical_depth_contract,
    load_empirical_runtime_release,
)
from .dinocular_backbone import BackendSpec, build_backbone, extract_features


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CheckpointLoadError(RuntimeError):
    """Raised when a checkpoint does not exactly match its declared schema."""


class CheckpointHashError(CheckpointLoadError):
    """Raised before deserialization when the artifact hash is wrong."""


@dataclass(frozen=True)
class LoadAudit:
    """Immutable evidence from a successful fail-closed backbone load."""

    checkpoint_sha256: str
    checkpoint_key: Optional[str]
    state_prefix: str
    checkpoint_tensor_count: int
    loaded_tensor_count: int
    allowed_outside_prefix: tuple[str, ...]
    allowed_checkpoint_only: tuple[str, ...]
    allowed_model_only: tuple[str, ...]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint_hash(path: str | Path, expected_sha256: str) -> str:
    expected = str(expected_sha256).lower()
    if not SHA256_RE.fullmatch(expected):
        raise ValueError("checkpoint_sha256 must be exactly 64 lowercase hexadecimal characters")
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Dinocular checkpoint does not exist: {artifact}")
    actual = sha256_file(artifact)
    if actual != expected:
        raise CheckpointHashError(
            f"SHA-256 mismatch for {artifact}: expected {expected}, got {actual}"
        )
    return actual


def _load_checkpoint(path: str | Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _extract_mapping(checkpoint: Any, checkpoint_key: Optional[str]) -> Mapping[str, Any]:
    value = checkpoint
    if checkpoint_key:
        for component in checkpoint_key.split("."):
            if not isinstance(value, Mapping) or component not in value:
                raise CheckpointLoadError(
                    f"Checkpoint key {checkpoint_key!r} is absent at component {component!r}"
                )
            value = value[component]
    if not isinstance(value, Mapping):
        raise CheckpointLoadError(
            f"Checkpoint selection {checkpoint_key!r} is {type(value).__name__}, not a state mapping"
        )
    if not value:
        raise CheckpointLoadError(f"Checkpoint selection {checkpoint_key!r} is empty")
    bad = [key for key, tensor in value.items() if not isinstance(key, str) or not torch.is_tensor(tensor)]
    if bad:
        raise CheckpointLoadError(
            "Selected checkpoint state contains non-tensor or non-string entries; "
            f"first invalid keys: {bad[:5]}"
        )
    return value


def _select_prefix(
    state: Mapping[str, torch.Tensor], state_prefix: str
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    if not isinstance(state_prefix, str):
        raise TypeError("state_prefix must be a string")
    if state_prefix:
        selected = {
            key[len(state_prefix) :]: tensor
            for key, tensor in state.items()
            if key.startswith(state_prefix)
        }
        outside = tuple(sorted(key for key in state if not key.startswith(state_prefix)))
    else:
        selected = dict(state)
        outside = ()
    if not selected:
        raise CheckpointLoadError(
            f"No checkpoint tensors matched declared state_prefix {state_prefix!r}"
        )
    if "" in selected:
        raise CheckpointLoadError("state_prefix selected an empty parameter name")
    return selected, outside


def _map_stock_sunrgbd_state(
    selected: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Apply the sole documented public-checkpoint mapping and allowlist.

    The SUNRGBD segmentation checkpoint carries three output norms, while the
    stock inference backbone has only its final norm.  Norms 0/1 feed only the
    discarded segmentation decoder; norm 2 is mapped to the final inference
    norm.  No other checkpoint-only key is accepted here.
    """

    mapped: dict[str, torch.Tensor] = {}
    allowed_checkpoint_only = []
    for key, tensor in selected.items():
        if key.startswith("extra_norms.0.") or key.startswith("extra_norms.1."):
            allowed_checkpoint_only.append(key)
            continue
        if key.startswith("extra_norms.2."):
            target_key = "extra_norms." + key.removeprefix("extra_norms.2.")
        else:
            target_key = key
        if target_key in mapped:
            raise CheckpointLoadError(
                f"Public checkpoint mapping produced duplicate model key {target_key!r}"
            )
        mapped[target_key] = tensor
    return mapped, tuple(sorted(allowed_checkpoint_only))


def load_backbone_checkpoint(
    backbone: nn.Module,
    spec: BackendSpec,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    checkpoint_key: Optional[str],
    state_prefix: str,
    allowed_outside_prefixes: Sequence[str] = (),
    allowed_missing_keys: Sequence[str] = (),
) -> LoadAudit:
    """Verify, audit, and strictly load a backbone under its registered policy."""

    actual_sha256 = verify_checkpoint_hash(checkpoint_path, checkpoint_sha256)
    raw_checkpoint = _load_checkpoint(checkpoint_path)
    state = _extract_mapping(raw_checkpoint, checkpoint_key)
    selected, outside = _select_prefix(state, state_prefix)

    if spec.checkpoint_policy == "stock_sunrgbd":
        if allowed_outside_prefixes or allowed_missing_keys:
            raise CheckpointLoadError(
                "Stock SUNRGBD checkpoint exceptions are fixed and cannot be "
                "widened by configuration"
            )
        effective_outside_prefixes = ("decode_head.",)
        mapped, allowed_checkpoint_only = _map_stock_sunrgbd_state(selected)
        allowed_model_names = {
            "mask_token",
            "pred.decoder.0.weight",
            "pred.decoder.0.bias",
        }
    elif spec.checkpoint_policy == "exact":
        effective_outside_prefixes = tuple(allowed_outside_prefixes)
        if any(
            not isinstance(prefix, str) or not prefix
            for prefix in effective_outside_prefixes
        ):
            raise CheckpointLoadError(
                "allowed_outside_prefixes must contain only non-empty strings"
            )
        mapped = dict(selected)
        allowed_checkpoint_only = ()
        if any(not isinstance(key, str) or not key for key in allowed_missing_keys):
            raise CheckpointLoadError(
                "allowed_missing_keys must contain only non-empty parameter names"
            )
        requested_missing = set(allowed_missing_keys)
        disallowed_requested = sorted(requested_missing - {"mask_token"})
        if disallowed_requested:
            raise CheckpointLoadError(
                "Configuration attempted to permit unsupported target-backbone missing keys: "
                f"{disallowed_requested}"
            )
        allowed_model_names = requested_missing
    else:
        raise CheckpointLoadError(
            f"Unimplemented checkpoint policy {spec.checkpoint_policy!r}"
        )

    disallowed_outside = sorted(
        key
        for key in outside
        if not any(key.startswith(prefix) for prefix in effective_outside_prefixes)
    )
    if disallowed_outside:
        raise CheckpointLoadError(
            "Checkpoint contains tensors outside the declared backbone prefix: "
            f"{disallowed_outside}"
        )

    model_state = backbone.state_dict()
    model_names = set(model_state)
    mapped_names = set(mapped)
    unexpected = sorted(mapped_names - model_names)
    missing = sorted(model_names - mapped_names)
    disallowed_missing = sorted(set(missing) - allowed_model_names)
    allowed_model_only = tuple(sorted(set(missing) & allowed_model_names))
    if unexpected or disallowed_missing:
        raise CheckpointLoadError(
            "Checkpoint/backbone key audit failed: "
            f"missing={disallowed_missing}, unexpected={unexpected}, "
            f"documented_model_only={list(allowed_model_only)}, "
            f"documented_checkpoint_only={list(allowed_checkpoint_only)}"
        )

    shape_mismatches = []
    for key, tensor in mapped.items():
        if tuple(tensor.shape) != tuple(model_state[key].shape):
            shape_mismatches.append(
                (key, tuple(tensor.shape), tuple(model_state[key].shape))
            )
    if shape_mismatches:
        raise CheckpointLoadError(
            f"Checkpoint/backbone tensor shape audit failed: {shape_mismatches}"
        )

    # Fill only the explicit stock inference-only allowlist, then use PyTorch's
    # strict=True path.  Every parameter used by forward_encoder came from the
    # verified artifact; there is no partial target-backbone load.
    complete_state = dict(mapped)
    for key in allowed_model_only:
        complete_state[key] = model_state[key]
    backbone.load_state_dict(complete_state, strict=True)

    return LoadAudit(
        checkpoint_sha256=actual_sha256,
        checkpoint_key=checkpoint_key,
        state_prefix=state_prefix,
        checkpoint_tensor_count=len(state),
        loaded_tensor_count=len(mapped),
        allowed_outside_prefix=outside,
        allowed_checkpoint_only=allowed_checkpoint_only,
        allowed_model_only=allowed_model_only,
    )


class DinocularEncoder(nn.Module):
    """Frozen native-token RGB-D encoder with explicit artifact metadata.

    ``depth`` must already come from the declared complete-trajectory cache or
    the stateful planning-prefix producer.  This adapter intentionally has no
    isolated-frame depth fallback.
    """

    def __init__(
        self,
        backend: str,
        factory: str,
        checkpoint_path: str,
        checkpoint_sha256: str,
        checkpoint_key: Optional[str],
        state_prefix: str,
        feature_key: str = "x_norm_patchtokens",
        input_size: int = 224,
        num_patches: int = 49,
        emb_dim: int = 512,
        frozen: bool = True,
        depth_mean: float = 0.48,
        depth_std: float = 0.28,
        depth_contract: Optional[Mapping[str, Any]] = None,
        allowed_outside_prefixes: Sequence[str] = (),
        allowed_missing_keys: Sequence[str] = (),
        depth_contract_status: str = "complete",
        native_depth_contract_path: Optional[str] = None,
        native_depth_contract_sha256: Optional[str] = None,
        selected_cache_producer_sha256: Optional[str] = None,
        selected_cache_environment: Optional[str] = None,
        empirical_depth_contract_path: Optional[str] = None,
        empirical_depth_contract_sha256: Optional[str] = None,
        empirical_runtime_release_path: Optional[str] = None,
        empirical_runtime_release_sha256: Optional[str] = None,
        depth_input_mode: str = "native_v1",
        empirical_zero_intervention: bool = False,
        neutralize_depth_at_encoder_input: bool = False,
        name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.backend = str(backend)
        self.factory = str(factory)
        self.feature_key = str(feature_key)
        self.input_size = int(input_size)
        self.num_patches = int(num_patches)
        self.emb_dim = int(emb_dim)
        self.frozen = bool(frozen)
        self.name = name or f"dinocular_{self.backend}_{self.factory}"
        self.requires_depth = True
        self.depth_contract = dict(depth_contract or {})
        self.checkpoint_path = str(checkpoint_path)
        self.checkpoint_sha256 = str(checkpoint_sha256).lower()
        self.checkpoint_key = checkpoint_key
        self.state_prefix = state_prefix
        self.allowed_outside_prefixes = tuple(allowed_outside_prefixes)
        self.allowed_missing_keys = tuple(allowed_missing_keys)
        self.depth_contract_status = str(depth_contract_status)
        self.depth_input_mode = str(depth_input_mode)
        self.empirical_zero_intervention = bool(empirical_zero_intervention)
        self.neutralize_depth_at_encoder_input = bool(
            neutralize_depth_at_encoder_input
        )
        self._encoder_boundary_hooks: list[
            Callable[["DinocularEncoder", torch.Tensor, torch.Tensor], None]
        ] = []
        self.native_depth_contract = None
        self.empirical_depth_contract = None
        self.cache_binding = None
        if (native_depth_contract_path is None) != (
            native_depth_contract_sha256 is None
        ):
            raise DepthContractError(
                "native depth contract path and SHA-256 must be supplied together"
            )
        if (empirical_depth_contract_path is None) != (
            empirical_depth_contract_sha256 is None
        ):
            raise DepthContractError(
                "empirical depth contract path and SHA-256 must be supplied together"
            )
        if (empirical_runtime_release_path is None) != (
            empirical_runtime_release_sha256 is None
        ):
            raise DepthContractError(
                "empirical runtime release path and SHA-256 must be supplied together"
            )
        if native_depth_contract_path is not None and empirical_depth_contract_path is not None:
            raise DepthContractError("native and empirical depth contracts are mutually exclusive")
        if empirical_depth_contract_path is not None:
            if self.depth_input_mode != "empirical_lossy_cache_v1":
                raise DepthContractError("empirical contract requires empirical_lossy_cache_v1 mode")
            if self.neutralize_depth_at_encoder_input:
                raise DepthContractError("native neutralization is forbidden for empirical input")
            if empirical_runtime_release_path is None:
                raise DepthContractError(
                    "empirical mode requires an independently accepted runtime release"
                )
            canonical_empirical = load_empirical_depth_contract(
                empirical_depth_contract_path,
                str(empirical_depth_contract_sha256),
                expected_checkpoint_sha256=self.checkpoint_sha256,
            )
            empirical_release = load_empirical_runtime_release(
                empirical_runtime_release_path,
                str(empirical_runtime_release_sha256),
                contract=canonical_empirical,
            )
            empirical = load_empirical_depth_contract(
                empirical_depth_contract_path,
                str(empirical_depth_contract_sha256),
                expected_checkpoint_sha256=self.checkpoint_sha256,
                runtime_resolver=empirical_release.resolver,
            )
            configured_checkpoint = Path(self.checkpoint_path)
            if configured_checkpoint != empirical.checkpoint_path:
                raise DepthContractError(
                    "configured checkpoint path differs from the runtime resolver"
                )
            empirical_release.resolver.validate_open_path(
                configured_checkpoint, artifact="empirical checkpoint"
            )
            checkpoint_contract = empirical.manifest["checkpoint"]
            configured = {
                "backend": self.backend,
                "factory": self.factory,
                "checkpoint_key": self.checkpoint_key,
                "state_prefix": self.state_prefix,
                "feature_key": self.feature_key,
                "input_shape": [1, self.input_size, self.input_size],
            }
            mismatches = {
                key: (checkpoint_contract.get(key), value)
                for key, value in configured.items()
                if checkpoint_contract.get(key) != value
            }
            if mismatches:
                raise DepthContractError(
                    f"empirical contract/checkpoint configuration mismatch: {mismatches}"
                )
            self.empirical_depth_contract = empirical
            self.depth_contract = dict(empirical.manifest)
            self.depth_contract_status = "complete"
            self.empirical_depth_contract_path = str(empirical.path)
            self.empirical_depth_contract_sha256 = empirical.sha256
            self.empirical_runtime_release_path = str(empirical_release.path)
            self.empirical_runtime_release_sha256 = empirical_release.sha256
            self.empirical_runtime_mode = empirical_release.mode
            self.selected_cache_producer_sha256 = empirical.producer_sha256
            self.selected_cache_environment = "pusht"
            self.native_depth_contract_path = None
            self.native_depth_contract_sha256 = None
        elif native_depth_contract_path is not None:
            if self.depth_input_mode != "native_v1" or self.empirical_zero_intervention:
                raise DepthContractError("native contract requires unchanged native_v1 semantics")
            native = load_native_depth_contract(
                native_depth_contract_path,
                str(native_depth_contract_sha256),
                expected_checkpoint_sha256=self.checkpoint_sha256,
            )
            checkpoint_contract = native.manifest["checkpoint"]
            configured = {
                "backend": self.backend,
                "factory": self.factory,
                "checkpoint_key": self.checkpoint_key,
                "state_prefix": self.state_prefix,
            }
            mismatches = {
                key: (checkpoint_contract.get(key), value)
                for key, value in configured.items()
                if checkpoint_contract.get(key) != value
            }
            if mismatches:
                raise DepthContractError(
                    f"native contract/checkpoint configuration mismatch: {mismatches}"
                )
            self.native_depth_contract = native
            if selected_cache_producer_sha256 is None or selected_cache_environment is None:
                raise DepthContractError(
                    "native depth contract requires selected producer and environment"
                )
            self.cache_binding = native.binding_for(
                selected_cache_producer_sha256, selected_cache_environment
            )
            self.depth_contract = dict(native.manifest)
            self.depth_contract_status = "complete"
            depth_mean = 0.0
            depth_std = 1.0
            self.native_depth_contract_path = str(native.path)
            self.native_depth_contract_sha256 = native.sha256
            self.selected_cache_producer_sha256 = self.cache_binding.producer_sha256
            self.selected_cache_environment = self.cache_binding.environment
            self.empirical_depth_contract_path = None
            self.empirical_depth_contract_sha256 = None
            self.empirical_runtime_release_path = None
            self.empirical_runtime_release_sha256 = None
            self.empirical_runtime_mode = None
        else:
            if selected_cache_producer_sha256 is not None or selected_cache_environment is not None:
                raise DepthContractError(
                    "selected cache producer/environment requires a native depth contract"
                )
            if self.depth_input_mode != "native_v1" or self.empirical_zero_intervention:
                raise DepthContractError("empirical mode requires an explicit empirical contract")
            self.native_depth_contract_path = None
            self.native_depth_contract_sha256 = None
            self.empirical_depth_contract_path = None
            self.empirical_depth_contract_sha256 = None
            self.empirical_runtime_release_path = None
            self.empirical_runtime_release_sha256 = None
            self.empirical_runtime_mode = None
            self.selected_cache_producer_sha256 = None
            self.selected_cache_environment = None
        if self.neutralize_depth_at_encoder_input and self.native_depth_contract is None:
            raise DepthContractError(
                "neutral depth requires an explicit complete native depth contract"
            )
        if self.depth_contract_status not in {"complete", "missing_from_gate0"}:
            raise ValueError(
                "depth_contract_status must be 'complete' or 'missing_from_gate0'"
            )

        if self.input_size <= 0 or self.input_size % 32 != 0:
            raise ValueError("DFormerv2 input_size must be a positive multiple of stride 32")
        native_patches = (self.input_size // 32) ** 2
        if self.feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
            expected_patches = native_patches
        elif self.feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
            expected_patches = 1
        else:
            raise ValueError(
                "feature_key must be 'x_norm_patchtokens' or 'x_norm_clstoken', "
                f"got {self.feature_key!r}"
            )
        if self.num_patches != expected_patches:
            raise ValueError(
                f"num_patches={self.num_patches} disagrees with {self.feature_key} "
                f"at input_size={self.input_size}; expected {expected_patches}"
            )
        if not torch.isfinite(torch.tensor([depth_mean, depth_std])).all() or depth_std <= 0:
            raise ValueError("depth_mean/depth_std must be finite and depth_std must be positive")

        self.backbone, self.backend_spec = build_backbone(self.backend, self.factory)
        expected_emb_dim = self.backend_spec.factories[self.factory]
        if self.emb_dim != expected_emb_dim:
            raise ValueError(
                f"emb_dim={self.emb_dim} disagrees with registered {self.backend}/{self.factory} "
                f"dimension {expected_emb_dim}"
            )

        self.register_buffer(
            "rgb_mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "depth_mean", torch.tensor(float(depth_mean), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "depth_std", torch.tensor(float(depth_std), dtype=torch.float32), persistent=False
        )
        if self.empirical_depth_contract is not None:
            cache_minimum, cache_maximum = 0.0, 1.0
            neutral_depth, neutral_mask = 0.0, 0.0
            affine_scale, affine_offset = 1.0, 0.0
            clip_minimum, clip_maximum = 0.0, 1.0
            self.depth_interpolation = "identity_224x224"
        elif self.native_depth_contract is None:
            cache_minimum, cache_maximum = 0.0, 1.0
            neutral_depth, neutral_mask = 0.0, 0.0
            affine_scale, affine_offset = 1.0, 0.0
            clip_minimum, clip_maximum = 0.0, 1.0
            self.depth_interpolation = "identity_224x224"
        else:
            assert self.cache_binding is not None
            cache_minimum = self.cache_binding.wire_minimum
            cache_maximum = self.cache_binding.wire_maximum
            neutral_depth, neutral_mask = 0.0, 1.0
            affine_scale = self.cache_binding.raw_metric_scale
            affine_offset = self.cache_binding.raw_metric_offset
            clip_minimum = self.cache_binding.raw_metric_minimum
            clip_maximum = self.cache_binding.raw_metric_maximum
            self.depth_interpolation = self.cache_binding.interpolation
        self.register_buffer(
            "depth_cache_range",
            torch.tensor([cache_minimum, cache_maximum], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "neutral_normalized_depth",
            torch.tensor(neutral_depth, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "neutral_validity_mask",
            torch.tensor(neutral_mask, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "cache_to_native_affine",
            torch.tensor([affine_scale, affine_offset], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "raw_metric_range",
            torch.tensor([clip_minimum, clip_maximum], dtype=torch.float32),
            persistent=False,
        )
        self.input_metadata = {
            "checkpoint_sha256": self.checkpoint_sha256,
            "backend": self.backend,
            "factory": self.factory,
            "feature_key": self.feature_key,
            "input_size": self.input_size,
            "num_patches": self.num_patches,
            "emb_dim": self.emb_dim,
            "native_depth_contract_sha256": self.native_depth_contract_sha256,
            "empirical_depth_contract_sha256": self.empirical_depth_contract_sha256,
            "depth_input_mode": self.depth_input_mode,
            "empirical_adapter_id": (
                EMPIRICAL_ADAPTER_ID
                if self.empirical_depth_contract is not None
                else None
            ),
            "empirical_zero_intervention": self.empirical_zero_intervention,
            "selected_cache_producer_sha256": self.selected_cache_producer_sha256,
            "selected_cache_environment": self.selected_cache_environment,
        }

        self.load_audit = load_backbone_checkpoint(
            backbone=self.backbone,
            spec=self.backend_spec,
            checkpoint_path=self.checkpoint_path,
            checkpoint_sha256=self.checkpoint_sha256,
            checkpoint_key=self.checkpoint_key,
            state_prefix=self.state_prefix,
            allowed_outside_prefixes=self.allowed_outside_prefixes,
            allowed_missing_keys=self.allowed_missing_keys,
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(not self.frozen)
        if self.frozen:
            self.eval()

    @staticmethod
    def _validate_finite_range(
        tensor: torch.Tensor, low: float, high: float, label: str
    ) -> None:
        if not torch.is_floating_point(tensor):
            raise TypeError(f"{label} must be a floating-point tensor, got {tensor.dtype}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{label} contains NaN or infinity")
        tolerance = 1e-4
        minimum = float(tensor.detach().amin())
        maximum = float(tensor.detach().amax())
        if minimum < low - tolerance or maximum > high + tolerance:
            raise ValueError(
                f"{label} range [{minimum}, {maximum}] is outside declared [{low}, {high}]"
            )

    def preprocess_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"RGB must have shape [B,3,H,W], got {tuple(rgb.shape)}")
        self._validate_finite_range(rgb, -1.0, 1.0, "RGB")
        recovered = (rgb + 1.0) * 0.5
        mean = self.rgb_mean.to(device=rgb.device, dtype=rgb.dtype)
        std = self.rgb_std.to(device=rgb.device, dtype=rgb.dtype)
        return (recovered - mean) / std

    def preprocess_depth(self, depth: torch.Tensor) -> torch.Tensor:
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        if depth.ndim != 4 or depth.shape[1] != 1:
            raise ValueError(
                f"Depth gray must have shape [B,H,W] or [B,1,H,W], got {tuple(depth.shape)}"
            )
        depth_range = self.depth_cache_range.detach().cpu().tolist()
        self._validate_finite_range(
            depth, float(depth_range[0]), float(depth_range[1]), "Depth gray"
        )
        if self.native_depth_contract is not None:
            affine = self.cache_to_native_affine.to(
                device=depth.device, dtype=depth.dtype
            )
            depth = depth * affine[0] + affine[1]
            raw_range = self.raw_metric_range.to(
                device=depth.device, dtype=depth.dtype
            )
            self._validate_finite_range(
                depth, float(raw_range[0]), float(raw_range[1]), "Raw metric depth"
            )
            return depth
        mean = self.depth_mean.to(device=depth.device, dtype=depth.dtype)
        std = self.depth_std.to(device=depth.device, dtype=depth.dtype)
        return (depth - mean) / std

    def prepare_depth_encoder_input(
        self,
        depth: torch.Tensor,
        depth_validity_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode raw metric depth, then apply the exact zero boundary intervention."""

        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        if depth.ndim != 4 or depth.shape[1] != 1:
            raise ValueError(
                f"Depth gray must have shape [B,H,W] or [B,1,H,W], got {tuple(depth.shape)}"
            )
        if depth_validity_mask is None:
            if self.native_depth_contract is not None:
                raise ValueError(
                    "native depth contract requires obs['depth_validity_mask']"
                )
            depth_validity_mask = torch.ones_like(depth)
        elif depth_validity_mask.ndim == 3:
            depth_validity_mask = depth_validity_mask.unsqueeze(1)
        if tuple(depth_validity_mask.shape) != tuple(depth.shape):
            raise ValueError(
                "depth validity mask shape differs from depth: "
                f"{tuple(depth_validity_mask.shape)} versus {tuple(depth.shape)}"
            )
        if self.empirical_depth_contract is not None:
            if not torch.is_floating_point(depth_validity_mask):
                raise TypeError("empirical payload-presence mask must be floating point")
            if not torch.isfinite(depth_validity_mask).all() or not torch.all(
                depth_validity_mask == 1.0
            ):
                raise ValueError(
                    "empirical payload-presence mask must be exact all-ones after validation"
                )
            adapted, boundary_mask, diagnostics = apply_empirical_depth_adapter(
                depth,
                zero_intervention=self.empirical_zero_intervention,
            )
            self.last_empirical_adapter_diagnostics = diagnostics
            return adapted, boundary_mask
        target_size = (self.input_size, self.input_size)
        if self.depth_interpolation == "identity_224x224":
            if tuple(depth.shape[-2:]) != target_size:
                raise ValueError(
                    f"identity depth interpolation requires spatial size {target_size}, "
                    f"got {tuple(depth.shape[-2:])}"
                )
        elif self.depth_interpolation == "bilinear_align_corners_false":
            depth = F.interpolate(
                depth, size=target_size, mode="bilinear", align_corners=False
            )
            depth_validity_mask = F.interpolate(
                depth_validity_mask, size=target_size, mode="nearest"
            )
        else:
            raise RuntimeError(
                f"unsupported manifest depth interpolation {self.depth_interpolation!r}"
            )
        if not torch.is_floating_point(depth_validity_mask):
            raise TypeError("depth validity mask must be floating point")
        self._validate_finite_range(
            depth_validity_mask, 0.0, 1.0, "Depth validity mask"
        )
        if not torch.all(
            (depth_validity_mask == 0.0) | (depth_validity_mask == 1.0)
        ):
            raise ValueError("depth validity mask must contain only exact 0 and 1")

        if self.native_depth_contract is not None and not torch.all(
            depth_validity_mask == 1.0
        ):
            raise ValueError(
                "native cache payload validation must reject invalid values before the encoder"
            )
        raw_metric_depth = self.preprocess_depth(depth)
        if self.neutralize_depth_at_encoder_input:
            raw_metric_depth = torch.zeros_like(raw_metric_depth)
        return raw_metric_depth, depth_validity_mask

    def register_encoder_boundary_hook(
        self,
        hook: Callable[["DinocularEncoder", torch.Tensor, torch.Tensor], None],
    ) -> Callable[[], None]:
        """Observe the exact depth and validity tensors passed at the encoder boundary."""

        if not callable(hook):
            raise TypeError("encoder boundary hook must be callable")
        self._encoder_boundary_hooks.append(hook)

        def remove() -> None:
            if hook in self._encoder_boundary_hooks:
                self._encoder_boundary_hooks.remove(hook)

        return remove

    def forward(
        self,
        rgb: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
        depth_validity_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.depth_contract_status != "complete":
            raise RuntimeError(
                "Student checkpoint depth contract is missing from Gate 0; forward and "
                "scientific runs are blocked until producer/model/units/temporal/scale/"
                "normalization metadata is supplied"
            )
        if depth is None:
            raise ValueError(
                "DinocularEncoder requires cached trajectory depth or stateful planning-prefix depth; "
                "isolated-frame/zero-depth fallback is forbidden"
            )
        if rgb.shape[0] != depth.shape[0]:
            raise ValueError(
                f"RGB/depth batch sizes differ: {rgb.shape[0]} versus {depth.shape[0]}"
            )
        if tuple(rgb.shape[-2:]) != (self.input_size, self.input_size):
            raise ValueError(
                f"RGB spatial size must be {(self.input_size, self.input_size)}, "
                f"got {tuple(rgb.shape[-2:])}"
            )
        rgb_normalized = self.preprocess_rgb(rgb)
        depth_normalized, effective_validity_mask = self.prepare_depth_encoder_input(
            depth, depth_validity_mask
        )
        for hook in tuple(self._encoder_boundary_hooks):
            hook(self, depth_normalized, effective_validity_mask)
        features = extract_features(
            self.backbone, self.backend_spec, rgb_normalized, depth_normalized
        )
        if self.feature_key not in features:
            raise RuntimeError(
                f"Backend {self.backend!r} did not return configured feature key "
                f"{self.feature_key!r}; available={sorted(features)}"
            )
        tokens = features[self.feature_key]
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(1)
        expected_shape = (rgb.shape[0], self.num_patches, self.emb_dim)
        if tuple(tokens.shape) != expected_shape:
            raise RuntimeError(
                f"Encoder output shape {tuple(tokens.shape)} does not match metadata {expected_shape}"
            )
        return tokens

    def train(self, mode: bool = True) -> "DinocularEncoder":
        if self.frozen:
            super().train(False)
        else:
            super().train(mode)
        return self
