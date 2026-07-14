import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.layers import DropPath, trunc_normal_
from typing import Tuple
from collections import OrderedDict

class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor):
        """
        input shape (b c h w)
        """
        x = x.permute(0, 2, 3, 1).contiguous()  # (b h w c)
        x = self.norm(x)  # (b h w c)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class DepthEmbed(nn.Module):
    def __init__(self, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(1, embed_dim // 2, 3, 2, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, 1, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.BatchNorm2d(embed_dim),
        )
        self.drop = nn.Dropout(p=0.3)

    def forward(self, depth):
        x = self.proj(depth)
        x = x.permute(0, 2, 3, 1)
        x = self.drop(x)
        return x

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, 3, 2, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, 1, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.BatchNorm2d(embed_dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).permute(0, 2, 3, 1)
        return x


class DWConv2d(nn.Module):
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        """
        input (b h w c)
        """
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class PatchMerging(nn.Module):
    """
    Patch Merging Layer.
    """

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        """
        x: B H W C
        """
        x = x.permute(0, 3, 1, 2).contiguous()  # (b c h w)
        x = self.reduction(x)  # (b oc oh ow)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (b oh ow oc)
        return x


def angle_transform(x, sin, cos):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    return (x * cos) + (torch.stack([-x2, x1], dim=-1).flatten(-2) * sin)


class GeoPriorGen(nn.Module):
    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # note operator precedence: "/ head_dim // 2" floored every exponent to 0 and
        # initialised all frequencies to 1.0 instead of a geometric spectrum
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim // 2, 1).float() / (self.head_dim // 2)))

        self.freqs_x = nn.Parameter(inv_freq.clone())
        self.freqs_y = nn.Parameter(inv_freq.clone())
        self.freqs_z = nn.Parameter(inv_freq.clone())
        
        self.weight = nn.Parameter(torch.tensor(1.0))

        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads)
        )
        self.register_buffer("decay", decay)

    def generate_pos_decay(self, H: int, W: int):
        """
        generate 2d decay mask, the result is (HW)*(HW)
        H, W are the numbers of patches at each column and row
        """
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)
        mask = grid[:, None, :] - grid[None, :, :]
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None]
        return mask
    
    def generate_1d_decay(self, l: int):
        """
        generate 1d decay mask, the result is l*l
        """
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None]
        return mask

    def forward(self, HW_tuple: Tuple[int], depth_map, split_or_not=False):
        """
        depth_map: depth patches
        HW_tuple: (H, W)
        H * W == l
        """
        H, W = HW_tuple
        device = depth_map.device

        y_pos = torch.arange(H, device=device).float().unsqueeze(1).repeat(1, W)
        x_pos = torch.arange(W, device=device).float().unsqueeze(0).repeat(H, 1)

        z_pos = F.interpolate(depth_map, size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
        z_pos = z_pos * self.weight

        angles_x = torch.einsum('hw, d -> hwd', x_pos, self.freqs_x).unsqueeze(0)
        angles_y = torch.einsum('hw, d -> hwd', y_pos, self.freqs_y).unsqueeze(0)

        angles_z = torch.einsum('bhw, d -> bhwd', z_pos, self.freqs_z)

        angle = angles_x + angles_y + angles_z
        angle = angle.repeat_interleave(2, dim=-1).unsqueeze(1)
        
        sin = angle.sin()
        cos = angle.cos()
        
        if split_or_not:
            mask_h = self.generate_1d_decay(HW_tuple[0])
            mask_w = self.generate_1d_decay(HW_tuple[1])

            mask_h = mask_h.unsqueeze(0).unsqueeze(2)
            mask_w = mask_w.unsqueeze(0).unsqueeze(2)

            geo_prior = ((sin, cos), (mask_h, mask_w))
        else:
            mask = self.generate_pos_decay(HW_tuple[0], HW_tuple[1])
            geo_prior = ((sin, cos), mask)
        
        return geo_prior


class Decomposed_GSA(nn.Module):
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim**-0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)

        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def forward(self, x: torch.Tensor, rel_pos):
        bsz, h, w, _ = x.size()

        (sin, cos), (mask_h, mask_w) = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k = k * self.scaling
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # (b n h w d1)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # (b n h w d1)

        qr = angle_transform(q, sin, cos)
        kr = angle_transform(k, sin, cos)

        qr_w = qr.transpose(1, 2)
        kr_w = kr.transpose(1, 2)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)
        qk_mat_w = qk_mat_w + mask_w.transpose(1, 2)
        qk_mat_w = torch.softmax(qk_mat_w, -1)
        v = torch.matmul(qk_mat_w, v)

        qr_h = qr.permute(0, 3, 1, 2, 4)
        kr_h = kr.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 2, 1, 4)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)
        qk_mat_h = qk_mat_h + mask_h.transpose(1, 2)
        qk_mat_h = torch.softmax(qk_mat_h, -1)
        output = torch.matmul(qk_mat_h, v)

        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)
        output = output + lepe
        output = self.out_proj(output)
        return output

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)


class Full_GSA(nn.Module):
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim**-0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def forward(self, x: torch.Tensor, rel_pos):
        """
        x: (b h w c)
        rel_pos: mask: (n l l)
        """
        bsz, h, w, _ = x.size()
        (sin, cos), mask = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k = k * self.scaling
        q = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        k = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)

        qr = angle_transform(q, sin, cos)
        kr = angle_transform(k, sin, cos)

        qr = qr.flatten(2, 3)
        kr = kr.flatten(2, 3)
        vr = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        vr = vr.flatten(2, 3)
        qk_mat = qr @ kr.transpose(-1, -2)
        qk_mat = qk_mat + mask
        qk_mat = torch.softmax(qk_mat, -1)
        output = torch.matmul(qk_mat, vr)
        output = output.transpose(1, 2).reshape(bsz, h, w, -1)
        output = output + lepe
        output = self.out_proj(output)
        return output

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn=F.gelu,
        dropout=0.0,
        activation_dropout=0.0,
        layernorm_eps=1e-6,
        subln=False,
        subconv=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = nn.LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None
        self.dwconv = DWConv2d(ffn_dim, 3, 1, 1) if subconv else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x: torch.Tensor):
        """
        input shape: (b h w c)
        """
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        residual = x
        if self.dwconv is not None:
            x = self.dwconv(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = x + residual
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x


class RGBD_Block(nn.Module):
    def __init__(
        self,
        split_or_not: str,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        drop_path=0.0,
        layerscale=False,
        layer_init_values=1e-5,
        init_value=2,
        heads_range=4,
    ):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=1e-6)
        if split_or_not:
            self.Attention = Decomposed_GSA(embed_dim, num_heads)
        else:
            self.Attention = Full_GSA(embed_dim, num_heads)
        self.drop_path = DropPath(drop_path)
        # FFN
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.cnn_pos_encode = DWConv2d(embed_dim, 3, 1, 1)
        # the function to generate the geometry prior for the current block
        self.Geo = GeoPriorGen(embed_dim, num_heads, init_value, heads_range)

        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)



    def forward(self, x: torch.Tensor, x_e: torch.Tensor, split_or_not=False):
        x = x + self.cnn_pos_encode(x)
        b, h, w, d = x.size()

        geo_prior = self.Geo((h, w), x_e, split_or_not=split_or_not)

        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.Attention(self.layer_norm1(x), geo_prior))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.layer_norm2(x)))
        else:
            x = x + self.drop_path(self.Attention(self.layer_norm1(x), geo_prior))
            x = x + self.drop_path(self.ffn(self.layer_norm2(x)))
        return x


class BasicLayer(nn.Module):
    """
    A basic RGB-D layer in DFormerv2.
    """

    def __init__(
        self,
        embed_dim,
        out_dim,
        depth,
        num_heads,
        init_value: float,
        heads_range: float,
        ffn_dim=96.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        split_or_not=False,
        downsample: PatchMerging = None,
        layerscale=False,
        layer_init_values=1e-5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.split_or_not = split_or_not

        # build blocks
        self.blocks = nn.ModuleList(
            [
                RGBD_Block(
                    split_or_not,
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    drop_path[i] if isinstance(drop_path, list) else drop_path,
                    layerscale,
                    layer_init_values,
                    init_value=init_value,
                    heads_range=heads_range,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=embed_dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_e):
        for blk in self.blocks:
            x = blk(x, x_e, split_or_not=self.split_or_not)

        if self.downsample is not None:
            x_down = self.downsample(x)
            return x, x_down
        else:
            return x, x


class dformerv2(nn.Module):
    def __init__(
        self,
        embed_dims=[64, 128, 256, 512],
        depths=[2, 2, 8, 2],
        num_heads=[4, 4, 8, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[4, 4, 3, 3],
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        norm_eval=True,
        num_classes=1000,
        mask_ratio=0.4,
        layerscales=[False, False, False, False],
        layer_init_values=1e-6,
        out_indices=(1, 2, 3),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.patch_norm = patch_norm
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios
        self.norm_eval = norm_eval
        self.out_indices = out_indices

        self.mask_ratio = mask_ratio

        # patch embedding
        self.patch_embed = PatchEmbed(
            in_chans=3, embed_dim=embed_dims[0], norm_layer=norm_layer if self.patch_norm else None
        )

        # drop path rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                embed_dim=embed_dims[i_layer],
                out_dim=embed_dims[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                init_value=init_values[i_layer],
                heads_range=heads_ranges[i_layer],
                ffn_dim=int(mlp_ratios[i_layer] * embed_dims[i_layer]),
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                split_or_not=(i_layer != 3),
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                layerscale=layerscales[i_layer],
                layer_init_values=layer_init_values,
            )
            self.layers.append(layer)

        # self.extra_norms = nn.LayerNorm(embed_dims[-1])
        self.extra_norms = nn.ModuleList()
        for i in range(3):
            self.extra_norms.append(nn.LayerNorm(embed_dims[i + 1]))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dims[0]))

        self.geo_embed = DepthEmbed(
            embed_dim=embed_dims[0], norm_layer=norm_layer if self.patch_norm else None
        )

        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dims[0] * 2, embed_dims[0]),
            nn.LayerNorm(embed_dims[0])
        )

        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(embed_dims[1], 256, kernel_size=1), # Stage 1: 128 -> 256
            nn.Conv2d(embed_dims[2], 256, kernel_size=1), # Stage 2: 256 -> 256
            nn.Conv2d(embed_dims[3], 256, kernel_size=1)  # Stage 3: 512 -> 256
        ])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            # logger = get_root_logger()
            _state_dict = torch.load(pretrained)
            if "model" in _state_dict.keys():
                _state_dict = _state_dict["model"]
            if "state_dict" in _state_dict.keys():
                _state_dict = _state_dict["state_dict"]
            state_dict = OrderedDict()

            for k, v in _state_dict.items():
                if k.startswith("backbone."):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v
            print("load " + pretrained)
            self.load_state_dict(state_dict, strict=False)
            # load_checkpoint(self, pretrained, strict=False)
            # load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x, x_e, masks=None):
        """
        Returns normalized cls/patch tokens for self-supervised objectives
        (DINO/iBOT style).

        Output shapes:
            - x_norm_clstoken: (B, C)
            - x_norm_patchtokens: (B, N, C)
            - x_norm_patchtokens_hw: (B, H, W, C)
        """
        x = self.patch_embed(x)
        x_geo = self.geo_embed(x_e)
        x = self.fusion_proj(torch.cat([x, x_geo], dim=-1))

        if masks is not None:
            if masks.ndim == 2:
                b, n = masks.shape
                h_mask = int(math.sqrt(n))
                w_mask = n // h_mask
                masks_2d = masks.view(b, 1, h_mask, w_mask).float()
            elif masks.ndim == 3:
                masks_2d = masks.unsqueeze(1).float()
            else:
                raise RuntimeError(f"Unsupported masks shape: {masks.shape}")

            masks_up = F.interpolate(
                masks_2d,
                size=(x.shape[1], x.shape[2]),
                mode="nearest",
            ).permute(0, 2, 3, 1)
            x = x * (1.0 - masks_up) + self.mask_token.to(x.dtype) * masks_up

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, x = layer(x, x_e)
            if i in self.out_indices: # 1,2,3
                x_out = self.extra_norms[i - 1](x_out)
                out = x_out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        b, h, w, c = x_out.shape
        patch_tokens = x_out.reshape(b, h * w, c)
        cls_token = patch_tokens.mean(dim=1)
        
        return {
            "x_norm_clstoken": cls_token,
            "x_norm_patchtokens": patch_tokens,
            "x_norm_patchtokens_hw": x_out,
            "x_feature_maps": self.upsample(outs),
        }
    
    def forward_encoder(self, x, x_e):
        x = self.patch_embed(x)
        x_geo = self.geo_embed(x_e)
        x = self.fusion_proj(torch.cat([x, x_geo], dim=-1))
        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, x = layer(x, x_e)
            if i in self.out_indices: # 1,2,3
                x_out = self.extra_norms[i - 1](x_out)
                out = x_out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return out, outs
    
    def upsample(self, feature_list): # [Tensor[B,C,H,W],...,]
        H, W = feature_list[0].shape[2:]
        output = []
        for i, feature in enumerate(feature_list):
            feature = self.fpn_convs[i](feature)
            if i > 0:
                feature = F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=False)
            output.append(feature)
        
        return torch.cat(output, dim=1)
    
    def forward(self, x, x_e, is_training=False, return_dict=False, masks=None):
        if is_training or return_dict:
            return self.forward_features(x, x_e, masks=masks)

        # if student:
            # return output_last, self.upsample(feature_list)
        return None, None

def DFormerv2_S(pretrained=False, **kwargs):
    model = dformerv2(
        embed_dims=[64, 128, 256, 512],
        depths=[3, 4, 18, 4],
        num_heads=[4, 4, 8, 16],
        heads_ranges=[4, 4, 6, 6],
        **kwargs,
    )
    return model

def DFormerv2_B(pretrained=False, **kwargs):
    model = dformerv2(
        embed_dims=[80, 160, 320, 512],
        depths=[4, 8, 25, 8],
        num_heads=[5, 5, 10, 16],
        heads_ranges=[5, 5, 6, 6],
        layerscales=[False, False, True, True],
        layer_init_values=1e-6,
        **kwargs,
    )
    return model

def DFormerv2_L(pretrained=False, **kwargs):
    model = dformerv2(
        embed_dims=[112, 224, 448, 640],
        depths=[4, 8, 25, 8],
        num_heads=[7, 7, 14, 20],
        heads_ranges=[6, 6, 6, 6],
        layerscales=[False, False, True, True],
        layer_init_values=1e-6,
        **kwargs,
    )
    return model
