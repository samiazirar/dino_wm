import torch
from torch import nn

from models.visual_world_model import VWorldModel


class _FrozenEncoder(nn.Module):
    input_size = 8
    num_patches = 4
    emb_dim = 3
    latent_ndim = 2
    name = "metadata_only"

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)

    def forward(self, rgb, depth=None):
        rgb = self.bn(rgb)
        pooled = torch.nn.functional.adaptive_avg_pool2d(rgb, (2, 2))
        return pooled.flatten(2).transpose(1, 2)


class _IdentitySequence(nn.Module):
    emb_dim = 3

    def forward(self, x):
        return x


class _DistributedLikeWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def test_frozen_encoder_stays_eval_and_accepts_optional_depth():
    raw_encoder = _FrozenEncoder()
    encoder = _DistributedLikeWrapper(raw_encoder)
    model = VWorldModel(
        image_size=16,
        num_hist=1,
        num_pred=1,
        encoder=encoder,
        proprio_encoder=_IdentitySequence(),
        action_encoder=_IdentitySequence(),
        decoder=None,
        predictor=None,
        proprio_dim=3,
        action_dim=3,
        concat_dim=1,
        num_action_repeat=1,
        num_proprio_repeat=1,
        train_encoder=False,
        train_predictor=False,
        train_decoder=False,
    )
    model.train()
    assert encoder.training is False
    assert raw_encoder.training is False
    assert raw_encoder.bn.training is False

    obs = {
        "visual": torch.rand(2, 1, 3, 16, 16),
        "depth": torch.rand(2, 1, 8, 8),
        "proprio": torch.rand(2, 1, 3),
    }
    out = model.encode_obs(obs)
    assert out["visual"].shape == (2, 1, 4, 3)
