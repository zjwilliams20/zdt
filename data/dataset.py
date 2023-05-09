#!/usr/bin/env python

"""PyTorch interface to trajectory data"""

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    def __init__(self, datapath):
        loaded = np.load(datapath)
        X, U = loaded["X"], loaded["U"]
        XU = np.concatenate([X[:, :-1], U], axis=2)

        rollouts, horizon = XU.shape[:2]
        self.data = torch.from_numpy(XU.reshape(rollouts, horizon, -1))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, k):
        return self.data[k]


if __name__ == "__main__":
    dset = TrajectoryDataset("ztt-DoubleIntegrator2D_R10_H40_dt0.1.npz")
