
import torch
import torch.nn as nn


class SimMIM(nn.Module):
    def __init__(self, embed_dim, patch_size=4, encoder_stride=32):
        super().__init__()

        self.embed_dim = embed_dim
        self.encoder_stride = encoder_stride
        self.patch_size = patch_size

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.embed_dim,
                out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )

        # self.in_chans = self.encoder.in_chans # 

    def forward(self, x):
        x_rec = self.decoder(x)

        # mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
        # loss_recon = F.l1_loss(x, x_rec, reduction='none')
        # loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        return x_rec
