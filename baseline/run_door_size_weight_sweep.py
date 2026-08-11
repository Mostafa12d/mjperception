"""
Generalization check: does the SAME adaptive-impedance controller + RLS
estimator + fixed-amplitude torque dither (tuned on the big door in
run_door_adaptive_impedance.py) still work on a smaller, lighter/heavier door?

Nothing about the controller is retuned here (Kp, Kd, TAU_MAX, DITHER_AMP,
DITHER_FREQ, RLS_LAM, TRACE_P_THRESH, RAMP_DURATION all stay exactly as in
run_door_adaptive_impedance.py). Only the plant changes:
  - door_small.xml: half the width/height of door.xml's panel
  - density swept light / heavy (frictionloss/damping fixed to a small-hinge
    default, since those represent hinge hardware, not panel mass)

Produces:
  adaptive_small_light.mp4, adaptive_small_heavy.mp4
  adaptive_small_light_dither_log.csv, adaptive_small_heavy_dither_log.csv

Run:
    python3.10 run_door_size_weight_sweep.py
"""

from __future__ import annotations

from run_door_adaptive_impedance import run_condition, DITHER_AMP, TAU_MAX, KP, KD

SMALL_DOOR_PATH = "door_small.xml"
SMALL_HANDLE_DIST = 0.4
SMALL_FRICTIONLOSS = 0.6
SMALL_DAMPING = 0.05

CONFIGS = [
    ("small_light", 200.0),
    ("small_heavy", 1400.0),
]


def main() -> None:
    print("Door size/weight generalization sweep (excited condition only)")
    print(f"  controller UNCHANGED: Kp={KP} Kd={KD} TAU_MAX={TAU_MAX} "
          f"DITHER_AMP={DITHER_AMP}\n")

    results = {}
    for name, density in CONFIGS:
        r = run_condition(
            excited=True,
            video_path=f"adaptive_{name}.mp4",
            model_path=SMALL_DOOR_PATH,
            handle_dist=SMALL_HANDLE_DIST,
            density=density,
            frictionloss=SMALL_FRICTIONLOSS,
            damping=SMALL_DAMPING,
            log_path=f"adaptive_{name}_dither_log.csv",
        )
        results[name] = r

    print("--- Generalization verdict ---")
    for name, r in results.items():
        saturated = float((abs(r["tau_cmd"]) >= TAU_MAX - 1e-6).mean()) * 100.0
        print(
            f"  {name}: I_true={r['I_true']:.3f}  I_err_final={r['I_hat'][-1] - r['I_true']:+.3f} "
            f"({r['rel_err']:.1f}%)  mu_err_final={r['mu_hat'][-1] - r['mu_true']:+.3f}  "
            f"track_RMSE={r['rmse_track']:.4f} rad  backtrack={100 * r['frac_back']:.1f}%  "
            f"torque_saturated={saturated:.1f}%"
        )
        if r["frac_back"] > 0.0 or saturated > 1.0:
            print(f"    -> dither is straining this config (reversals and/or torque clipping)")
        else:
            print(f"    -> dither did not disrupt tracking; controller generalized cleanly")


if __name__ == "__main__":
    main()
