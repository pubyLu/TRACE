import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from blocks import BasicEncoder, Mlp
from dat_arch import ResidualGroup


class Rewarding(nn.Module):
    def __init__(self, in_channel=2, input_H=80, input_W=80, depth=2, drop_path_rate=0.1, drop_rate=0.,
                 attn_drop_rate=0.,):
        super(Rewarding, self).__init__()
        self.H, self.W = input_H, input_W
        self.in_channel = in_channel

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.ftn = BasicEncoder(in_channel, output_dim=64)
        self.att = ResidualGroup(
            dim=64, num_heads=8, reso=self.H, split_size=[8, 10], expansion_factor=2, qkv_bias=True, qk_scale=None,
            drop= drop_rate, attn_drop=attn_drop_rate, drop_paths=dpr, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            depth=depth, use_chk=False, resi_connection='1conv', rg_idx=0
        )
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp= Mlp(
            in_features=64,
            hidden_features=256,
            act_layer=approx_gelu,
            drop=0,
        )
        self.tail = nn.Conv2d(64, 1, kernel_size=1)
    def forward(self, x):
        x = self.ftn(x)
        x = torch.flatten(x, start_dim=2)
        x = x.permute(0, 2, 1)
        x = self.att(x, (self.H, self.W))
        x = self.mlp(x)
        x = x.permute(0, 2, 1)
        B, C, N = x.shape
        x = x.reshape(B, C, self.H, self.W)
        x = self.tail(x)
        x = F.sigmoid(x).reshape(-1, 1, self.H, self.W)
        return x

