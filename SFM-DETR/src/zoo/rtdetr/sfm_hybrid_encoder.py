"""SFM-DETR encoder; public baseline parameter names are preserved."""
import torch
from torch import nn
from torch.nn import functional as F
from ...core import register
from .hybrid_encoder import HybridEncoder
from .scwa import SCWA
from .mfa_neck import MFANeck


@register()
class SFMHybridEncoder(HybridEncoder):
    def __init__(self, in_channels=[128, 256, 512], feat_strides=[8, 16, 32],
                 hidden_dim=256, nhead=8, dim_feedforward=1024, dropout=0., enc_act='gelu',
                 use_encoder_idx=[2], num_encoder_layers=1, pe_temperature=10000,
                 expansion=.5, depth_mult=1., act='silu', eval_spatial_size=None,
                 version='v2', use_scwa=True, use_mfa=True, chunk_rows=16):
        super().__init__(in_channels, feat_strides, hidden_dim, nhead, dim_feedforward,
                         dropout, enc_act, use_encoder_idx, num_encoder_layers, pe_temperature,
                         expansion, depth_mult, act, eval_spatial_size, version)
        self.use_scwa, self.use_mfa = use_scwa, use_mfa
        # New modules must not change the random initialization of the decoder.
        with torch.random.fork_rng(devices=[]):
            self.scwa = nn.ModuleList([SCWA(hidden_dim, chunk_rows=chunk_rows) for _ in range(2)]) if use_scwa else nn.ModuleList()
            self.mfa = nn.ModuleList([MFANeck(hidden_dim, chunk_rows=chunk_rows) for _ in range(2)]) if use_mfa else nn.ModuleList()

    def forward(self, feats):
        if not self.use_scwa and not self.use_mfa:
            return super().forward(feats)
        projected = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        original = list(projected)
        preferences = [None, None]
        for i in range(2):
            if self.use_scwa:
                projected[i], preferences[i] = self.scwa[i](original[i], original[i+1])
            else:
                preferences[i] = original[i].new_full((original[i].shape[0], 4, *original[i].shape[-2:]), .25)
        if self.num_encoder_layers > 0:
            for i, index in enumerate(self.use_encoder_idx):
                h, w = projected[index].shape[-2:]
                src = projected[index].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos = self.build_2d_sincos_position_embedding(w, h, self.hidden_dim, self.pe_temperature).to(src.device)
                else:
                    pos = getattr(self, f'pos_embed{index}')
                projected[index] = self.encoder[i](src, pos_embed=pos).permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
        inner_outs = [projected[-1]]
        for idx in range(2, 0, -1):
            stage = 2-idx
            high = self.lateral_convs[stage](inner_outs[0])
            inner_outs[0] = high
            low = projected[idx-1]
            up = self.mfa[stage](low, high, preferences[idx-1]) if self.use_mfa else F.interpolate(high, scale_factor=2., mode='nearest')
            inner_outs.insert(0, self.fpn_blocks[stage](torch.cat([up, low], 1)))
        outs = [inner_outs[0]]
        for idx in range(2):
            outs.append(self.pan_blocks[idx](torch.cat([self.downsample_convs[idx](outs[-1]), inner_outs[idx+1]], 1)))
        return outs
