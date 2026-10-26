"""Single-grid morphology-aware frequency-guided cross-scale attention."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class MFANeck(nn.Module):
    def __init__(self, channels=256, dim=32, heads=4, chunk_rows=16):
        super().__init__()
        self.heads, self.dim, self.chunk_rows = heads, dim, chunk_rows
        self.geometry = nn.Sequential(nn.Conv2d(2*channels+4, 64, 1), nn.SiLU(), nn.Conv2d(64, 5, 1))
        nn.init.zeros_(self.geometry[-1].weight)
        nn.init.zeros_(self.geometry[-1].bias)
        with torch.no_grad():
            self.geometry[-1].bias[3:] = math.log((1-.25)/(3.-1.))
        self.q_proj = nn.Conv2d(channels+4, dim, 1)
        self.kv_proj = nn.Conv2d(channels, dim*2, 1)
        self.preference_bias = nn.Conv2d(4, heads*9, 1)
        self.position_bias = nn.Parameter(torch.zeros(heads, 2))
        self.out_proj = nn.Conv2d(dim, channels, 1, bias=False)
        self.eta = nn.Parameter(torch.tensor(1e-3))
        yy, xx = torch.meshgrid(torch.arange(-1, 2), torch.arange(-1, 2), indexing='ij')
        self.register_buffer('template', torch.stack([xx, yy], -1).reshape(9, 2).float())

    def _attend(self, q, kv, geometry, bias, centers):
        b, _, rows, w = q.shape
        hc, wc = kv.shape[-2:]
        d = self.dim // self.heads
        # Compute coordinates in fp32 so AMP cannot quantize pixel centers.
        with torch.autocast(device_type=q.device.type, enabled=False):
            g = geometry.float().permute(0, 2, 3, 1)
            delta = 2 * g[..., :2].tanh()
            angle = math.pi * g[..., 2].tanh()
            scale = .25 + 2.75 * g[..., 3:].sigmoid()
            base = scale.unsqueeze(-2) * self.template
            cos, sin = angle.cos().unsqueeze(-1), angle.sin().unsqueeze(-1)
            rx = cos*base[..., 0] - sin*base[..., 1]
            ry = sin*base[..., 0] + cos*base[..., 1]
            relative = torch.stack([rx, ry], -1) + delta.unsqueeze(-2)
            coords = centers + relative
            valid = ((coords[..., 0] >= 0) & (coords[..., 0] <= wc-1) &
                     (coords[..., 1] >= 0) & (coords[..., 1] <= hc-1))
            grid = (coords + .5) * coords.new_tensor([2/wc, 2/hc]) - 1
            sampled = F.grid_sample(kv.float(), grid.reshape(b, rows, w*9, 2), align_corners=False)
        sampled = sampled.reshape(b, 2, self.heads, d, rows, w, 9).to(q.dtype)
        keys, values = sampled[:, 0], sampled[:, 1]
        scores = (q.reshape(b, self.heads, d, rows, w).unsqueeze(-1)*keys).sum(2).float() * d**-.5
        scores = scores + bias.reshape(b, self.heads, 9, rows, w).permute(0, 1, 3, 4, 2)
        scores = scores + torch.einsum('bxykc,hc->bhxyk', relative, self.position_bias.float())
        valid = valid[:, None]
        any_valid = valid.any(-1, keepdim=True)
        scores = scores.masked_fill(~valid, float('-inf'))
        scores = torch.where(any_valid, scores, torch.zeros_like(scores))
        a = scores.softmax(-1) * valid
        context = (a.to(values.dtype).unsqueeze(2) * values).sum(-1)
        return context.reshape(b, self.dim, rows, w)

    def forward(self, fine, coarse, preference):
        b, _, h, w = fine.shape
        hc, wc = coarse.shape[-2:]
        up = F.interpolate(coarse, (h, w), mode='nearest')
        preference = preference.to(fine.dtype)
        geometry = self.geometry(torch.cat([fine, up, preference], 1))
        q = self.q_proj(torch.cat([fine, preference], 1))
        kv = self.kv_proj(coarse)
        bias = self.preference_bias(preference)
        yy, xx = torch.meshgrid(torch.arange(h, device=fine.device, dtype=torch.float32),
                                torch.arange(w, device=fine.device, dtype=torch.float32), indexing='ij')
        centers = torch.stack([(xx+.5)*wc/w-.5, (yy+.5)*hc/h-.5], -1)[None, :, :, None]
        chunks = []
        for start in range(0, h, self.chunk_rows):
            end = min(start+self.chunk_rows, h)
            args = (q[:, :, start:end], kv, geometry[:, :, start:end], bias[:, :, start:end], centers[:, start:end])
            if self.training and torch.is_grad_enabled():
                chunks.append(checkpoint(self._attend, *args, use_reentrant=False))
            else:
                chunks.append(self._attend(*args))
        return up + self.eta * self.out_proj(torch.cat(chunks, 2))
