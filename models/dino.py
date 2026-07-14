"""Pinned, offline DINOv2 encoder adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


DINOV2_VITS14_SHA256 = (
    "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DinoV2Encoder(nn.Module):
    """DINOv2 loaded only from an explicit local checkout and checkpoint."""

    def __init__(
        self,
        name: str,
        feature_key: str,
        repo_dir: str,
        weights_path: str,
        weights_sha256: Optional[str] = DINOV2_VITS14_SHA256,
        frozen: bool = True,
    ):
        super().__init__()
        self.name = name
        self.feature_key = feature_key
        self.frozen = frozen
        self.input_size = 196
        self.num_patches = 196 if feature_key == "x_norm_patchtokens" else 1

        repo = Path(repo_dir).expanduser().resolve()
        weights = Path(weights_path).expanduser().resolve()
        if not (repo / "hubconf.py").is_file():
            raise FileNotFoundError(f"Pinned DINOv2 checkout is missing hubconf.py: {repo}")
        if not weights.is_file():
            raise FileNotFoundError(f"Pinned DINOv2 checkpoint does not exist: {weights}")
        if weights_sha256 is not None:
            actual = _sha256(weights)
            if actual != weights_sha256:
                raise RuntimeError(
                    f"DINOv2 checkpoint SHA-256 mismatch: expected {weights_sha256}, "
                    f"got {actual} ({weights})"
                )

        # source='local' is deliberate: compute nodes must never access torch.hub.
        self.base_model = torch.hub.load(
            str(repo), name, source="local", pretrained=False
        )
        checkpoint = torch.load(weights, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Expected a state dict in {weights}, got {type(checkpoint)!r}")
        incompat = self.base_model.load_state_dict(checkpoint, strict=True)
        if incompat.missing_keys or incompat.unexpected_keys:
            raise RuntimeError(
                "Strict DINOv2 loading unexpectedly reported incompatibilities: "
                f"missing={incompat.missing_keys}, unexpected={incompat.unexpected_keys}"
            )

        self.emb_dim = self.base_model.num_features
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")
        self.patch_size = self.base_model.patch_size

        if self.frozen:
            self.requires_grad_(False)
            super().train(False)

    def train(self, mode: bool = True):
        super().train(False if self.frozen else mode)
        return self

    def forward(self, x: torch.Tensor, depth: Optional[torch.Tensor] = None):
        del depth  # RGB-only baseline intentionally ignores the common depth input.
        emb = self.base_model.forward_features(x)[self.feature_key]
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1)
        return emb
