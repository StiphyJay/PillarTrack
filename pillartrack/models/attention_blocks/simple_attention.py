import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath

class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        # self.act = nn.GELU()
        self.act = nn.LeakyReLU(negative_slope=0.01)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, nheads, attn_drop, proj_drop):
        super().__init__()
        head_dim = dim // nheads
        self.scale = head_dim ** -0.5
        self.nheads = nheads

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)


        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value):
        B, N, C = query.shape
        q = self.to_q(query)
        k = self.to_k(key)
        v = self.to_v(value)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = self.nheads), (q, k, v))

        attn = (q @ k.transpose(2, 1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, nheads, 
                attn_drop=0., drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.attn = Attention(in_dim, nheads, attn_drop, drop)
        self.ffn = FFN(in_dim, hidden_dim, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def with_pos(self, x, pos=None):
        return x if pos is None else x + pos

    def forward(self, x, t, pos=None):
        xn = self.norm1(x)
        xp = self.with_pos(xn, pos)
        x = x + self.drop_path(self.attn(xp, self.norm1(t), self.norm1(t)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x

class SelfBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, nheads, 
                attn_drop=0., drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.attn = Attention(in_dim, nheads, attn_drop, drop)
        self.ffn = FFN(in_dim, hidden_dim, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def with_pos(self, x, pos=None):
        return x if pos is None else x + pos

    def forward(self, x, pos=None):
        xn = self.norm1(x)
        xp = self.with_pos(xn, pos)
        x = x + self.drop_path(self.attn(xp, xp, xp))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x