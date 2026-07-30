"""Pinned DFormerv2 backbones used by :mod:`models.dinocular`.

The implementation files imported here are byte-for-byte copies from MThesis-wt
commit ``b99422df5115f769899b5432f5b2b81796d3a27c``:

* ``models/DFormerv2.py`` (SHA-256
  ``aaeb1992312ebc9738279edd984f8e6f0b238f6e3f6d344074256f515ec17b77``)
* ``models/DF2_DINO_rope_convs_de.py`` (SHA-256
  ``2dab217777185d19b144be22fafe80dc941fd0a67ae9a6e5fc5cd380b48ab1e6``)

Only the registry and inference dispatch below are integration code.  Keeping
the two implementations separate is deliberate: the public DFormerv2 weights
must never be partially loaded into the DINOcular backend.
"""

from dataclasses import dataclass
from types import ModuleType
from typing import Callable, Dict, Mapping

import torch
from torch import nn

from . import _vendor_df2_dino_rope_convs_de as _target
from . import _vendor_dformerv2_stock as _stock


MTHESIS_COMMIT = "b99422df5115f769899b5432f5b2b81796d3a27c"


@dataclass(frozen=True)
class BackendSpec:
    """A closed backend definition; no import-by-string fallback is allowed."""

    module: ModuleType
    factories: Mapping[str, int]
    output_kind: str
    checkpoint_policy: str


BACKEND_REGISTRY: Dict[str, BackendSpec] = {
    "dformerv2_stock": BackendSpec(
        module=_stock,
        factories={"DFormerv2_S": 512, "DFormerv2_B": 512, "DFormerv2_L": 640},
        output_kind="stock_feature_map",
        checkpoint_policy="stock_sunrgbd",
    ),
    "df2_dino_rope_convs_de": BackendSpec(
        module=_target,
        factories={"DFormerv2_S": 512, "DFormerv2_B": 512, "DFormerv2_L": 640},
        output_kind="dino_encoder_feature_map",
        checkpoint_policy="exact",
    ),
}


def get_backend_spec(backend: str) -> BackendSpec:
    """Return a registered backend or fail closed."""

    try:
        return BACKEND_REGISTRY[backend]
    except KeyError as exc:
        choices = ", ".join(sorted(BACKEND_REGISTRY))
        raise ValueError(f"Unknown Dinocular backend {backend!r}; expected one of: {choices}") from exc


def build_backbone(backend: str, factory: str) -> tuple[nn.Module, BackendSpec]:
    """Construct a checkpoint-faithful backbone with stochastic depth disabled."""

    spec = get_backend_spec(backend)
    if factory not in spec.factories:
        choices = ", ".join(sorted(spec.factories))
        raise ValueError(
            f"Factory {factory!r} is not registered for backend {backend!r}; "
            f"expected one of: {choices}"
        )
    factory_fn: Callable[..., nn.Module] = getattr(spec.module, factory)
    backbone = factory_fn(pretrained=False, drop_path_rate=0.0, mask_ratio=0.0)

    nonzero_drop_paths = []
    for name, module in backbone.named_modules():
        drop_prob = getattr(module, "drop_prob", None)
        if drop_prob is not None and float(drop_prob) != 0.0:
            nonzero_drop_paths.append((name, float(drop_prob)))
    if nonzero_drop_paths:
        raise RuntimeError(f"Backbone contains nonzero stochastic-depth modules: {nonzero_drop_paths}")
    return backbone, spec


def extract_features(
    backbone: nn.Module,
    spec: BackendSpec,
    rgb: torch.Tensor,
    depth: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    """Run the backend's provenance-matched inference path."""

    if spec.output_kind == "stock_feature_map":
        feature_map, mask = backbone.forward_encoder(rgb, depth)
        if mask is not None:
            raise RuntimeError("Stock DFormerv2 inference unexpectedly applied random masking")
        if feature_map.ndim != 4:
            raise RuntimeError(
                "Stock DFormerv2 forward_encoder must return [B,C,H,W], "
                f"got {tuple(feature_map.shape)}"
            )
        patch_tokens = feature_map.flatten(2).transpose(1, 2).contiguous()
        return {
            "x_norm_patchtokens": patch_tokens,
            "x_norm_clstoken": patch_tokens.mean(dim=1),
        }

    if spec.output_kind == "dino_encoder_feature_map":
        feature_map, _intermediate_maps = backbone.forward_encoder(rgb, depth)
        if feature_map.ndim != 4:
            raise RuntimeError(
                "DF2_DINO_rope_convs_de.forward_encoder must return [B,C,H,W], "
                f"got {tuple(feature_map.shape)}"
            )
        patch_tokens = feature_map.flatten(2).transpose(1, 2).contiguous()
        return {
            "x_norm_patchtokens": patch_tokens,
            "x_norm_clstoken": patch_tokens.mean(dim=1),
        }

    if spec.output_kind == "dino_feature_dict":
        features = backbone.forward_features(rgb, depth)
        if not isinstance(features, Mapping):
            raise RuntimeError(
                "DF2_DINO_rope_convs_de.forward_features must return a mapping, "
                f"got {type(features).__name__}"
            )
        return features

    raise RuntimeError(f"Unsupported registered output kind: {spec.output_kind!r}")
