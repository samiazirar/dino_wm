"""Strict RGB-D encoder adapter for stock DFormerv2 and DINOcular backends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Optional, Sequence

import torch
from torch import nn

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
) -> LoadAudit:
    """Verify, audit, and strictly load a backbone under its registered policy."""

    actual_sha256 = verify_checkpoint_hash(checkpoint_path, checkpoint_sha256)
    raw_checkpoint = _load_checkpoint(checkpoint_path)
    state = _extract_mapping(raw_checkpoint, checkpoint_key)
    selected, outside = _select_prefix(state, state_prefix)

    if spec.checkpoint_policy == "stock_sunrgbd":
        if allowed_outside_prefixes:
            raise CheckpointLoadError(
                "Stock SUNRGBD outside-prefix policy is fixed to decode_head.* and "
                "cannot be widened by configuration"
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
        allowed_model_names: set[str] = set()
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

        self.load_audit = load_backbone_checkpoint(
            backbone=self.backbone,
            spec=self.backend_spec,
            checkpoint_path=self.checkpoint_path,
            checkpoint_sha256=self.checkpoint_sha256,
            checkpoint_key=self.checkpoint_key,
            state_prefix=self.state_prefix,
            allowed_outside_prefixes=self.allowed_outside_prefixes,
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
        self._validate_finite_range(depth, 0.0, 1.0, "Depth gray")
        mean = self.depth_mean.to(device=depth.device, dtype=depth.dtype)
        std = self.depth_std.to(device=depth.device, dtype=depth.dtype)
        return (depth - mean) / std

    def forward(self, rgb: torch.Tensor, depth: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        if tuple(depth.shape[-2:]) != (self.input_size, self.input_size):
            raise ValueError(
                f"Depth spatial size must be {(self.input_size, self.input_size)}, "
                f"got {tuple(depth.shape[-2:])}"
            )

        rgb_normalized = self.preprocess_rgb(rgb)
        depth_normalized = self.preprocess_depth(depth)
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
