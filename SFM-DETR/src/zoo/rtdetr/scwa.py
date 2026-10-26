"""Semantic-conditioned, signed Haar band/neighborhood attention (SFM-DETR)."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class SCWA(nn.Module):
    def __init__(self, channels=256, active=64, dim=32, heads=4, chunk_rows=16):
        super().__init__()
        self.heads, self.dim, self.chunk_rows = heads, dim, chunk_rows
        self.x_proj = nn.Conv2d(channels, active, 1)
        self.semantic_proj = nn.Conv2d(channels, active, 1)
        self.band_norm = nn.GroupNorm(1, active)
        self.q_proj = nn.Conv2d(2 * active, dim, 1)
        self.k_proj = nn.Conv2d(2 * active, dim, 1)
        self.v_proj = nn.Conv2d(active, dim, 1)
        self.band_embedding = nn.Parameter(torch.zeros(4, dim))
        self.relative_bias = nn.Parameter(torch.zeros(heads, 4, 9))
        self.dwconv = nn.Conv2d(active, active, 3, padding=1, groups=active)
        self.out_proj = nn.Conv2d(active + dim, channels, 1)
        self.gamma = nn.Parameter(torch.tensor(1e-3))
        # L, Hh (horizontal differences), Hv (vertical differences), Hd.
        self.register_buffer('haar', torch.tensor([
            [[1., 1.], [1., 1.]], [[1., -1.], [1., -1.]],
            [[1., 1.], [-1., -1.]], [[1., -1.], [-1., 1.]]
        ]).unsqueeze(1) / 2)

    def _attend(self, q, k, v, valid):
        b, _, rows, w = q.shape
        d = self.dim // self.heads
        # k/v include a one-pixel halo; only materialize a spatial tile.
        k = F.unfold(k.flatten(0, 1), 3).reshape(b, 4, self.heads, d, 9, rows, w)
        v = F.unfold(v.flatten(0, 1), 3).reshape(b, 4, self.heads, d, 9, rows, w)
        q = q.reshape(b, self.heads, d, rows, w)
        scores = torch.einsum('bhdxy,bshdnxy->bhsnxy', q, k) * d ** -0.5
        scores = scores.float() + self.relative_bias[None, :, :, :, None, None]
        scores = scores.masked_fill(~valid[:, None, None], float('-inf'))
        a = scores.flatten(2, 3).softmax(2).reshape(b, self.heads, 4, 9, rows, w)
        z = torch.einsum('bhsnxy,bshdnxy->bhdxy', a.to(v.dtype), v)
        preference = a.sum(3).mean(1)
        return z.reshape(b, self.dim, rows, w), preference

    def forward(self, fine, deeper):
        x = self.x_proj(fine)
        b, c, h, w = x.shape
        s = F.interpolate(self.semantic_proj(deeper), (h, w), mode='bilinear', align_corners=False)
        q = self.q_proj(torch.cat([x, s], 1))
        bands = F.conv2d(F.pad(x.reshape(b*c, 1, h, w), (0, 1, 0, 1), mode='replicate'), self.haar)
        bands = bands.reshape(b, c, 4, h, w).transpose(1, 2).reshape(b*4, c, h, w)
        # Identical group normalization for each band; signed values are retained.
        normalized = self.band_norm(bands)
        with torch.autocast(device_type=x.device.type, enabled=False):
            magnitude = bands.float().abs()
            padded = F.pad(magnitude, (2, 2, 2, 2), mode='replicate')
            mean = F.avg_pool2d(padded, 5, stride=1)
            variance = (F.avg_pool2d(padded.square(), 5, stride=1) - mean.square()).clamp_min(0)
            relative = ((magnitude - mean) / (variance + 1e-6).sqrt()).reshape(b, 4, c, h, w)
            relative = torch.cat([torch.zeros_like(relative[:, :1]), relative[:, 1:]], 1).reshape(b*4, c, h, w)
        k = self.k_proj(torch.cat([normalized, relative.to(normalized.dtype)], 1)).reshape(b, 4, self.dim, h, w)
        k = k + self.band_embedding[None, :, :, None, None].to(k.dtype)
        v = self.v_proj(normalized).reshape(b, 4, self.dim, h, w)
        k, v = F.pad(k, (1, 1, 1, 1)), F.pad(v, (1, 1, 1, 1))
        mask = F.pad(torch.ones(1, 1, h, w, device=x.device), (1, 1, 1, 1))
        contexts, preferences = [], []
        for start in range(0, h, self.chunk_rows):
            end = min(start + self.chunk_rows, h)
            valid = F.unfold(mask[:, :, start:end+2], 3).reshape(1, 9, end-start, w).bool()
            args = (q[:, :, start:end], k[:, :, :, start:end+2], v[:, :, :, start:end+2], valid)
            if self.training and torch.is_grad_enabled():
                z, pref = checkpoint(self._attend, *args, use_reentrant=False)
            else:
                z, pref = self._attend(*args)
            contexts.append(z)
            preferences.append(pref)
        z = torch.cat(contexts, 2)
        return fine + self.gamma * self.out_proj(torch.cat([self.dwconv(x), z], 1)), torch.cat(preferences, 2)
