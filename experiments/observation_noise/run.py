"""Where does RLS lose to the learned method as the encoder gets worse?

    python3.10 -m experiments.observation_noise.run

The project's strongest claim is that the crossover between classical system
identification and the learned method is SENSING, not physics: RLS must form
``qddot = d(qdot)/dt``, so encoder noise lands directly in its regressor, while
the learned predictor never differentiates.

This is the experiment that only became a configuration change after the refactor.
Previously the sensor pipeline lived inside ``mismatch/``, was wired only to the
door stream builder, and could not be pointed at another predictor or mechanism
without new code. Here the sweep is a list of observation models.

CAVEAT, carried from CURRENT_SYSTEM.md E.2: the published crossover figure was
measured on the doors-only predictor while the geometry and UKF results use the
all-families predictor (open item B1). This spec records its checkpoint, so the
discrepancy is visible in ``spec.json`` rather than buried. It does NOT resolve it
-- that is a research decision, deliberately left open.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments import ExperimentSpec, run_experiment
from mechanics import IdentityObservation, JointSensor, MethodConfig
from mechanics.metrics import write_csv

# Encoder noise, in radians. The published crossover sits near 14-bit resolution
# over the door's 2.26 rad travel, i.e. sigma ~ 1.4e-4 rad.
NOISE_LEVELS = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)


def spec_for(sigma: float, out_root: str) -> ExperimentSpec:
    obs = (IdentityObservation() if sigma == 0.0
           else JointSensor(theta_sigma=sigma))
    return ExperimentSpec(
        question=(
            "As encoder noise increases, at what point does the learned mechanics "
            "belief overtake RLS on one-step prediction of unseen doors?"),
        data="data/door_mechanics.npz",
        split="heldout_door",
        exclude_near_limit=True,
        observation=obs,
        checkpoint="runs/latent_mechanics/base/best.pt",
        methods=("no-adaptation", "gradient", "ukf", "rls-5p"),
        method_config=MethodConfig(init="medoid"),
        seeds=(0,),
        out_dir=f"{out_root}/sigma_{sigma:g}",
        metrics=("angle_nrmse_final", "us_per_update"),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-objects", type=int, default=4)
    ap.add_argument("--max-episodes", type=int, default=3)
    ap.add_argument("--out-dir", default="runs/experiments/observation_noise")
    args = ap.parse_args()

    all_rows = []
    for sigma in NOISE_LEVELS:
        spec = spec_for(sigma, args.out_dir)
        spec.max_objects = args.max_objects
        spec.max_episodes = args.max_episodes
        print(f"\n{'#' * 74}\n# encoder sigma = {sigma:g} rad\n{'#' * 74}")
        rows, _ = run_experiment(spec, verbose=True)
        for r in rows:
            r["theta_sigma"] = sigma
        all_rows += rows

    out = Path(args.out_dir)
    write_csv(out / "sweep.csv", all_rows)

    print("\n" + "=" * 74)
    print("CROSSOVER TABLE  (median tail nRMSE over objects; lower is better)")
    print("=" * 74)
    methods = list(dict.fromkeys(r["method"] for r in all_rows))
    print(f"  {'sigma [rad]':>12}" + "".join(f"{m:>16}" for m in methods))
    for sigma in NOISE_LEVELS:
        sub = [r for r in all_rows if r["theta_sigma"] == sigma]
        if not sub:
            continue
        cells = ""
        for m in methods:
            v = [r["angle_nrmse_final"] for r in sub if r["method"] == m]
            cells += f"{np.median(v):>16.4f}" if v else f"{'-':>16}"
        print(f"  {sigma:>12g}{cells}")

    print(f"\n  sweep -> {out / 'sweep.csv'}")
    print("\n  Reading: RLS should degrade far faster than the learned methods,")
    print("  because noise enters its regressor through a finite difference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
