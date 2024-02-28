#!/usr/bin/env python

"""Training script for ZTT"""

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from data.dataset import TrajectoryDataset


repopath = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--epochs", default=50, help="Number of epochs to train over"
    )
    parser.add_argument("datapath", help="Path to input dataset")
    args = parser.parse_args()

    dataset = TrajectoryDataset(args.datapath)
    loader = DataLoader(dataset, batch_size=1, num_workers=1)
    
    for sample in loader:
        print(sample.shape)

    for e in range(args.epochs):
        pass

if __name__ == "__main__":
    main()
