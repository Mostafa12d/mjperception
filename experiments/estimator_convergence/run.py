"""Does online adaptation beat the no-adaptation control, and does RLS still win?

    python3.10 -m experiments.estimator_convergence.run

This is the migration of Stage-2 Experiment 3 (``online/experiments.py``) onto the
shared core. Same protocol, same estimators, same prequential scoring -- but the
whole thing is now a spec plus a call, and for the first time the RLS baseline and
the UKF are built by the same code path (the audit found that no single site in
the old code built both).
"""

from __future__ import annotations

import argparse

from mechanics import IdentityObservation, MethodConfig
from experiments import ExperimentSpec, run_experiment

SPEC = ExperimentSpec(
    question=(
        "On unseen doors, does adapting the mechanics belief online beat holding "
        "it fixed -- and does classical RLS still win on a clean, well-modelled "
        "plant?"),

    # plant: the held-out door population, never seen during training
    data="data/door_mechanics.npz",
    split="heldout_door",
    exclude_near_limit=True,          # limit contact adds torque outside the action

    # observation: perfect proprioception. The sensing question is a separate
    # experiment (see experiments/observation_noise) precisely so this one is not
    # confounded by it.
    observation=IdentityObservation(),

    # predictor: the Stage-1 doors-only network, frozen
    checkpoint="runs/latent_mechanics/base/best.pt",

    # estimators. 'no-adaptation' is the control and is not optional.
    methods=("no-adaptation", "gradient", "ukf", "rls-5p", "rls-3p"),
    method_config=MethodConfig(
        init="medoid",   # a real, central training door; 'zero' is a hole in the cloud
        lr=0.03, window=32, lr_decay=3.0e-3,
    ),

    seeds=(0,),
    out_dir="runs/experiments/estimator_convergence",
    metrics=("angle_nrmse_final", "angle_rmse_final", "us_per_update",
             "belief_travel"),
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-objects", type=int, default=None)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    spec = SPEC
    if args.max_objects is not None:
        spec.max_objects = args.max_objects
    if args.max_episodes is not None:
        spec.max_episodes = args.max_episodes
    if args.out_dir:
        spec.out_dir = args.out_dir

    run_experiment(spec)
    print("\n  Reading: RLS is expected to win here. The simulated door is exactly")
    print("  linear-in-parameters, which is the regime RLS was built for. The")
    print("  learned method's claimed advantage is degraded sensing -- see")
    print("  experiments/observation_noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
