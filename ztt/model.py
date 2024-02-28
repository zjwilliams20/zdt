#!/usr/bin/env python

"""ZachTrajectoryTransformer Implementation

Notes
-----
With inspiration from:

- https://github.com/karpathy/nanoGPT/blob/master/model.py

Some notation:

- n: state dimension
- u: action dimension
- d: token dimension

"""

import pathlib
import tomllib

import torch
from torch import nn


N_HORIZON_MAX = 50


def read_config():
    CONFPATH = pathlib.Path(__file__).parent / "config.toml"
    with open(CONFPATH, "rb") as file:
        return tomllib.load(file)


class ZTT(nn.Module):
    def __init__(self, n, m, d, n_heads, dropout, n_layers):
        super().__init__()
        self.n = n
        self.m = m
        self.d = d

        self.t_embed = nn.Embedding(N_HORIZON_MAX, d)
        self.state_proj = nn.Linear(n, d)
        self.control_proj = nn.Linear(m, d)
        self.embed_ln = nn.LayerNorm(d)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=d,
            dropout=dropout,
            norm_first=True,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, n_layers)

    def forward(self, x):
        # x.shape === (batch, horizon, n + m)
        b, T, _ = x.shape
        device = x.device

        t = torch.arange(H, device=device)
        ...


if __name__ == "__main__":
    config = read_config()["model"]
    ztt = ZTT(
        config["n_embed"], config["n_heads"], config["dropout"], config["n_layers"]
    )

    B = 10
    H = 50
    D = 6 + 3
    x = torch.randn(B, H * D)
    y = ztt(x)
