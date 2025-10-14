from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_self = nn.Linear(config.n_embd, config.n_embd) # for self atom interaction
        self.n_head = config.n_head
        self.n_embd = config.n_embd
    
    def forward(self, x, dist_matrix, coulomb_matrix):
        B, T, C = x.size() # nmolecules (B), natoms (T), channels (C)
        # C (channels) is nh*hs ("number of heads" times "head size")
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))) # (B, nh, T, T)
        att = torch.sigmoid(att)
        # att = (torch.square(att) - 1) / math.sqrt(2)
        # att = - att * dist_matrix.unsqueeze(1) # Hadamard product (B, nh, T, T) * (B, 1, T, T) -> (B, nh, T, T)
        # att = F.softmax(att, dim=-1)
        att = att * coulomb_matrix.unsqueeze(1) # Hadamard product (B, nh, T, T) * (B, 1, T, T) -> (B, nh, T, T)
        y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, nh, hs) -> (B, T, C)
        y = y + self.c_self(x) 
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, dist_matrix, coulomb_matrix):
        x = self.attn(self.ln_1(x), dist_matrix, coulomb_matrix)
        x = self.mlp(self.ln_2(x))
        return x

@dataclass
class TAEConfig:
    natoms_type: int = 15
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 32


class TAEmodel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.L = 2.0 # distance scaling
        self.transformer = nn.ModuleDict(dict(
            ate = nn.Embedding(config.natoms_type, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # self.mlp = MLP(config)
        self.lm_head = nn.Linear(config.n_embd, 1, bias=False)
    
    def compute_coulomb_matrix(self, pos):
        a = 1.0 / (torch.cdist(pos, pos, p=2)/self.L + 1e-8) # (B, T, T)
        i = torch.arange(a.size()[-1])
        a[:, i, i] = 0.0
        # a[torch.logical_and(a>=0, a<=1e-8)] = 0.0
        return a # (B, T, T)

    def forward(self, x):  
        (atoms_idx, pos) = x
        B, T = atoms_idx.size()
        dist_matrix = torch.cdist(pos, pos, p=2)/self.L # (B, T, T)
        coulomb_matrix = self.compute_coulomb_matrix(pos) # (B, T, T)
        x = self.transformer.ate(atoms_idx) # (B, T, n_embd)
        for block in self.transformer.h:
            x = block(x, dist_matrix, coulomb_matrix)
        x = self.transformer.ln_f(x) # (B, T, n_embd)
        # x = self.mlp(x)
        tae = torch.sum(self.lm_head(x).squeeze(-1), dim=-1) # sum((B, T)) -> (B)

        return tae
