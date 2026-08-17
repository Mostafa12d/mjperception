"""SANITY EXPERIMENT -- the smallest run in which the whole system is visible.

    python3.10 -m experiments.sanity_one_door.run

    Question: starting from a deliberately WRONG mechanics belief about one unseen
              door, does the innovation carry enough information to correct it?

The goal is not performance. The goal is that every arrow in the architecture can
be pointed at, named, and printed. The script therefore does two things:

  1. Narrates the first few timesteps, printing every variable in the chain
     observation -> prediction -> innovation -> belief update -> new belief,
     with its shape, its units and the file that produced it.
  2. Runs the full stream for three methods and plots what happened.

Everything here uses only ``mechanics/`` -- no stage-specific package.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from latent_mechanics.dataset import DoorTransitionDataset
from mechanics import (
    IdentityObservation,
    MethodConfig,
    Workspace,
    build_method,
    run,
    transitions_from_dataset,
)
from mechanics.metrics import score
from mechanics.types import Belief

DATA = "data/door_mechanics.npz"
CHECKPOINT = "runs/latent_mechanics/base/best.pt"
OUT = Path("runs/experiments/sanity_one_door")

# How wrong to start. 3.0 = three standard deviations along the latent's dominant
# axis, which the geometry investigation identified as mechanical scale -- so this
# is roughly "believe the door is far heavier than it is".
WRONG_BY = 3.0

METHODS = ("no-adaptation", "gradient", "ukf")


# --------------------------------------------------------------------------
# 1. narrate one timestep
# --------------------------------------------------------------------------

def narrate(method, transitions, n_steps: int = 3) -> None:
    """Print every variable in the loop, for the first few timesteps."""
    print("\n" + "=" * 74)
    print("ONE TIMESTEP, END TO END")
    print("=" * 74)
    est, pred = method.estimator, method.predictor
    rep = pred.representation

    print(f"  predictor      {pred.name:24s} {type(pred).__name__}")
    print(f"  representation {rep.name:24s} dim={rep.dim}")
    print(f"  estimator      {est.name:24s} {type(est).__name__}")
    print(f"  innovation is measured in: {pred.measurement_space}")

    belief = est.initialize()
    for k in range(n_steps):
        tr = transitions[k]
        print(f"\n  --- step {k} " + "-" * 58)
        print(f"  observation      obs      = [{tr.obs[0]:+.6f} rad, "
              f"{tr.obs[1]:+.6f} rad/s]        <- ObservationModel")
        print(f"  action           a        = [{float(np.ravel(tr.action)[0]):+.4f} N*m]"
              f"                        <- excitation, ZOH")
        print(f"  belief (before)  x        = {np.array2string(belief.mean[:4], precision=3)}"
              f"{' ...' if belief.dim > 4 else ''}  ({rep.name}, dim {belief.dim})")

        prediction = pred.predict(tr.obs, tr.action, belief)
        print(f"  PREDICTED next   s_hat    = [{prediction[0]:+.6f} rad, "
              f"{prediction[1]:+.6f} rad/s]        <- Predictor.predict")

        print(f"  ACTUAL next      s_next   = [{tr.next_obs[0]:+.6f} rad, "
              f"{tr.next_obs[1]:+.6f} rad/s]        <- the plant")

        err = prediction - tr.target
        print(f"  reported error   s_hat-s  = [{err[0]:+.3e},     {err[1]:+.3e}]"
              f"      <- what the metrics score")

        belief, rec = est.update(belief, tr)
        print(f"  INNOVATION       nu       = "
              f"[{rec.innovation[0]:+.3e},     {rec.innovation[1]:+.3e}]"
              f"      <- what moved the belief ({rec.innovation_space})")
        print(f"  belief (after)   x'       = {np.array2string(belief.mean[:4], precision=3)}"
              f"{' ...' if belief.dim > 4 else ''}")

    print("\n  Note: the reported error and the innovation are DIFFERENT quantities")
    print("  in different spaces. That is real, not a bug -- see CURRENT_SYSTEM.md B.6.")


# --------------------------------------------------------------------------
# 2. figures
# --------------------------------------------------------------------------

def open_loop(predictor, belief_mean, transitions, horizon: int) -> np.ndarray:
    """Roll the predictor forward from one state, feeding predictions back in.

    One-step error flatters any smooth function over 20 ms -- the whole trajectory
    overlays the truth and you see nothing. An open-loop rollout is where a wrong
    belief actually shows up, which is why this is the panel that gets plotted.
    """
    b = Belief(mean=belief_mean)
    s = np.asarray(transitions[0].obs, dtype=np.float32)
    traj = [s.copy()]
    for t in transitions[:horizon]:
        s = predictor.predict(s, t.action, b)
        traj.append(np.asarray(s, dtype=np.float32))
    return np.stack(traj)


def figure(traces: dict, methods: dict, transitions, ws: Workspace, ds,
           object_id: int, out: Path) -> Path:
    """Three panels: true vs predicted, error over time, belief over time."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 11))
    truth = np.stack([t.target for t in transitions])
    colors = {"no-adaptation": "tab:red", "gradient": "tab:blue", "ukf": "tab:green"}

    # -- panel 1: true behaviour vs open-loop prediction ---------------------
    ax = axes[0]
    # a horizon inside the first episode, so no reset discontinuity intrudes
    horizon = min(120, len(transitions) - 1)
    true_traj = np.concatenate([truth[:horizon, 0][None, :],
                                truth[horizon - 1: horizon, 0][None, :]], axis=1)[0]
    ax.plot(np.arange(len(true_traj)), true_traj, "k-", lw=2.6,
            label="true $\\theta$", zorder=5)

    for name, m in methods.items():
        if name == "no-adaptation":
            continue
        start = open_loop(m.predictor, traces[name].beliefs[0], transitions, horizon)
        final = open_loop(m.predictor, traces[name].beliefs[-1], transitions, horizon)
        ax.plot(start[:, 0], "--", lw=1.5, color=colors.get(name), alpha=0.65,
                label=f"{name}: rollout under the WRONG initial belief")
        ax.plot(final[:, 0], "-", lw=1.7, color=colors.get(name),
                label=f"{name}: rollout under the CONVERGED belief")

    ax.set_ylabel("hinge angle [rad]")
    ax.set_xlabel(f"open-loop rollout step (horizon {horizon})")
    ax.set_title(f"Sanity experiment -- unseen door {object_id}, belief deliberately "
                 f"wrong by {WRONG_BY}$\\sigma$ along PC1\n"
                 f"Open-loop rollout: this is where a wrong belief shows")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    # -- panel 2: does the prediction error come down? -----------------------
    ax = axes[1]
    for name, tr in traces.items():
        ax.semilogy(tr.rolling_rmse(0, 200), lw=1.6, color=colors.get(name), label=name)
    ref = ws.reference_rmse["angle"]
    if np.isfinite(ref):
        ax.axhline(ref, color="grey", ls="--", lw=1.2,
                   label=f"Stage-1 reference ({ref:.1e})")
    ax.set_ylabel("rolling angle RMSE [rad]")
    ax.set_xlabel("interaction step")
    ax.set_title("Prequential one-step error (rolling window 200). "
                 "The control is the flat line.")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, which="both")

    # -- panel 3: the belief itself ------------------------------------------
    ax = axes[2]
    centred = ws.train_latents - ws.train_latents.mean(axis=0)
    pcs = np.linalg.svd(centred, full_matrices=False)[2][:2]
    proj = centred @ pcs.T
    inertia = np.array([ds.params_for_door(i)["I_hinge"]
                        for i in range(len(ws.train_latents))])
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=inertia, cmap="viridis", s=42,
                    alpha=0.75, edgecolors="none", label="training doors")
    plt.colorbar(sc, ax=ax, label="training door $I_{hinge}$ [kg m$^2$]")

    for name, tr in traces.items():
        if name == "no-adaptation":
            continue
        full = build_map(tr, methods[name])
        p = (full - ws.train_latents.mean(axis=0)) @ pcs.T
        ax.plot(p[:, 0], p[:, 1], "-", lw=1.4, color=colors.get(name), alpha=0.9)
        ax.plot(p[0, 0], p[0, 1], "X", ms=13, color=colors.get(name),
                mec="k", label=f"{name}: start (wrong)")
        ax.plot(p[-1, 0], p[-1, 1], "*", ms=18, color=colors.get(name),
                mec="k", label=f"{name}: converged")
    ax.set_xlabel("latent PC1  (= mechanical scale)")
    ax.set_ylabel("latent PC2")
    ax.set_title("Mechanics belief over time, in the frozen PCA chart")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def build_map(trace, method) -> np.ndarray:
    """Belief trajectory in full 16-D latent coordinates, whatever chart it used.

    Uses the method's OWN representation rather than a globally-configured basis,
    so a chart fitted on a different checkpoint cannot silently be applied here.
    """
    rep = method.predictor.representation
    return np.stack([np.asarray(rep.to_predictor(x)).reshape(-1)
                     for x in trace.beliefs])


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--object-id", type=int, default=None)
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--max-episodes", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("SANITY EXPERIMENT")
    print("=" * 74)
    print("  Question: starting from a deliberately wrong mechanics belief about")
    print("            one unseen door, does the innovation carry enough")
    print("            information to correct it?")

    ws = Workspace.load(args.checkpoint, stage="experiment:sanity_one_door")
    ds = DoorTransitionDataset(args.data, "heldout_door", exclude_near_limit=True)
    train_ds = DoorTransitionDataset(args.data, "train", exclude_near_limit=True)
    oid = args.object_id if args.object_id is not None else int(ds.door_ids[0])

    transitions, bounds = transitions_from_dataset(
        ds, oid, IdentityObservation(), max_episodes=args.max_episodes,
        exclude_near_limit=True)

    p = ds.params_for_door(oid)
    print(f"\n  door {oid}: I={p['I_hinge']:.2f} kg m^2, mu={p['frictionloss']:.2f} N m, "
          f"b={p['damping']:.2f}, k={p['stiffness']:.2f}")
    print(f"  {len(transitions)} transitions over {len(bounds)} episodes "
          f"at {1 / ds.dt_model:.0f} Hz")

    # THE DELIBERATELY WRONG PRIOR -- the whole point of the experiment
    z_wrong = ws.wrong_init("medoid", scale=WRONG_BY)
    z_right = ws.init_latent("medoid")
    print(f"  starting belief is {np.linalg.norm(z_wrong - z_right):.2f} away from the "
          f"medoid prior ({WRONG_BY}sigma along PC1)")

    cfg = MethodConfig(init_vector=z_wrong)

    traces, methods, rows = {}, {}, []
    for name in METHODS:
        m = build_method(name, ws, ds.dt_model, cfg)
        methods[name] = m
        if name == "gradient":
            narrate(m, transitions, n_steps=3)
        tr = run(m.estimator, m.predictor, transitions, object_id=oid,
                 boundaries=bounds, verify_frozen=True)
        traces[name] = tr
        rows.append({"method": m.name, **score(tr, transitions)})

    print("\n" + "=" * 74)
    print("RESULT")
    print("=" * 74)
    print(f"  {'method':16s} {'start RMSE':>12} {'final RMSE':>12} "
          f"{'nRMSE final':>12} {'us/update':>11} {'belief moved':>13}")
    n0 = max(1, len(transitions) // 20)
    for name, tr in traces.items():
        r = next(x for x in rows if x["method"].startswith(name[:3]))
        print(f"  {name:16s} {tr.rmse(0, last=n0):>12.3e} "
              f"{tr.final_rmse(0):>12.3e} {r['angle_nrmse_final']:>12.4f} "
              f"{r['us_per_update']:>11.1f} {tr.belief_travel:>13.3f}")

    ctrl = traces["no-adaptation"].final_rmse(0)
    print()
    for name, tr in traces.items():
        if name == "no-adaptation":
            continue
        g = ctrl / tr.final_rmse(0) if tr.final_rmse(0) > 0 else float("nan")
        verdict = "helped" if g > 1.05 else ("HURT" if g < 0.95 else "no effect")
        print(f"  {name:16s} {g:6.2f}x vs the no-adaptation control   -> {verdict}")

    path = figure(traces, methods, transitions, ws, train_ds, oid, out / "sanity.png")
    print(f"\n  figure -> {path}")

    from mechanics.metrics import write_csv
    write_csv(out / "results.csv", rows)
    print(f"  table  -> {out / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
