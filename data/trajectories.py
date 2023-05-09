#!/usr/bin/env python

"""Script to generate trajectory data according to some dynamical model"""

import argparse
from pathlib import Path
from time import strftime

import numpy as np

import dpilqr


model_map = {
    "DoubleIntegrator2D": dpilqr.DoubleIntDynamics4D,
    "Car3D": dpilqr.CarDynamics3D,
    "Unicycle4D": dpilqr.UnicycleDynamics4D,
    "Quadcopter6D": dpilqr.QuadcopterDynamics6D,
    "Bike5D": dpilqr.BikeDynamics5D,
}


def random_trajectories(model, rollouts, horizon, scale):
    """Randomly generate trajectories for some dynamical model"""

    u_bounds = (-scale / 2, +scale / 2)
    U = np.random.uniform(*u_bounds, (rollouts, horizon - 1, model.n_u))
    X = np.zeros((rollouts, horizon, model.n_x))

    for Xr, Ur in zip(X, U):
        for t, u in enumerate(Ur):
            Xr[t + 1] = model(Xr[t], u)

    return X, U


def main():
    parser = argparse.ArgumentParser(
        description="Barebones random trajectory generation for basic linear and some "
        "non-linear systems"
    )
    parser.add_argument(
        "model", choices=model_map.keys(), help="Dynamical system to include"
    )
    parser.add_argument(
        "-R",
        "--rollouts",
        default=10,
        type=int,
        help="Number of rollouts for each model",
    )
    parser.add_argument(
        "-N",
        "--horizon",
        default=40,
        type=int,
        help="Simulation horizon",
    )
    parser.add_argument(
        "--dt",
        default=0.1,
        type=float,
        help="Simulation time step",
    )
    parser.add_argument(
        "--scale",
        "-s",
        default=0.1,
        type=float,
        help="Scaling for uniform random control input",
    )

    args = parser.parse_args()
    model = model_map[args.model](args.dt)
    outpath = Path(__file__).parent / strftime(
        f"ztt-{args.model}_R{args.rollouts}_H{args.horizon}_dt{args.dt}.npz"
    )

    X, U = random_trajectories(model, args.rollouts, args.horizon, args.scale)

    print(f"Saving trajectory data to file: {outpath}...")
    np.savez(outpath, X=X, U=U)


if __name__ == "__main__":
    main()
