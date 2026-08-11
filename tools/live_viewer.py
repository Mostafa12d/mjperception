"""
Live MuJoCo viewer for one articulated mechanism under the project's own excitation.

Read-only tool. It builds nothing new: the object comes from
``mechanisms.library.build_model``, the driving torque comes from
``mechanisms.library.scaled_profile`` (the family-aware wrapper around
``excitation.sample_profile``), and the integrator loop applies the action the
same way ``mechanisms.rollout.simulate_mechanism`` does -- straight onto
``qfrc_applied[dof]``, no arm, no actuator, no contact.

Three things it shows, in increasing order of what they need:

  1. the mechanism moving, paced to wall-clock time;
  2. a live readout of joint position, velocity and the applied force/torque;
  3. optionally, the online belief update running against a frozen predictor.

The excitation is *identical* to what data generation produces. The per-episode
RNG seed follows ``rollout.rollout_mechanism``'s formula exactly, so
``--seed S --instance I`` reproduces the same torque stream that instance saw
when the training set was built. This matters: the point of the tool is to watch
the distribution the model was trained on, not a new signal that happens to look
similar.

TIMING. MuJoCo integrates at ``dyn.DT`` = 0.002 s (500 Hz). The learned model
predicts every ``sim.frame_skip`` = 10 steps, i.e. dt_model = 0.02 s (50 Hz).
The simulation is stepped at the fast rate; the belief update is fed at the slow
one. Feeding it faster would show the model something it has never seen.

Run (macOS needs mjpython for the interactive viewer):

    mjpython tools/live_viewer.py --family door
    mjpython tools/live_viewer.py --family drawer --speed 0.5 --duration 30

Without a window (physics + readout only, useful over ssh or in a check):

    python3.10 tools/live_viewer.py --family laptop --no-viewer --duration 5
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

# The project imports its modules relative to the repo root (``excitation.py``
# does ``import run_door_dynamics_validation``), so make sure the root is
# importable however this script was invoked.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import run_door_dynamics_validation as dyn  # noqa: E402
from latent_mechanics.config import ExperimentConfig, load_config  # noqa: E402
from latent_mechanics.mechanisms import library as lib  # noqa: E402
from latent_mechanics.mechanisms.rollout import near_limit_mask  # noqa: E402


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def in_repo_root():
    """Run a block with the repo root as cwd.

    ``FamilySpec.resolve_xml`` returns a bare ``"door.xml"`` for the four door
    families, because that asset lives at the repo root and is reused unchanged.
    MuJoCo then resolves it against the *current* directory, so building a door
    from anywhere else fails. Every other family resolves to an absolute path
    under ``mechanisms/assets`` and is unaffected. Scoped to model construction
    rather than applied process-wide, so the caller's cwd is never left changed.
    """
    prev = os.getcwd()
    os.chdir(_ROOT)
    try:
        yield
    finally:
        os.chdir(prev)


def units_for(joint_type: str) -> tuple[str, str, str]:
    """(position, velocity, force) unit labels for a joint type.

    The state interface deliberately does not rescale between revolute and
    prismatic joints (see ``library.py``'s module docstring), so the readout has
    to label which one it is showing or the numbers are ambiguous.
    """
    if joint_type == "prismatic":
        return "m", "m/s", "N"
    return "rad", "rad/s", "N*m"


# ---------------------------------------------------------------------------
# Step 3: the belief pipeline
# ---------------------------------------------------------------------------

@dataclass
class BeliefReadout:
    """What the belief module has to say right now, already formatted."""

    z_norm: float
    z_reduced: np.ndarray | None
    p_trace: float | None
    rmse_q: float
    rmse_v: float
    n_updates: int
    n_skipped: int


class BeliefRunner:
    """Drives an ``OnlineLatentAdaptor`` at the model's training cadence.

    Backend selection is deliberate rather than hardcoded. ``latent_mechanics.
    belief`` (the UKF branch) is not present on every branch of this repo; when
    it is missing, ``--belief auto`` falls back to the gradient-descent baseline
    in ``latent_mechanics.online.adaptor``, which is always there. The only
    visible difference is that the gradient adaptor's ``belief()["cov"]`` is
    ``None``, so there is no trace-of-P to display -- reported honestly as "n/a"
    rather than filled with a placeholder number.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        kind: str = "auto",
        init_name: str = "zero",
        lr: float = 0.03,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        from latent_mechanics.model import load_checkpoint
        from latent_mechanics.online.adaptor import GradientLatentAdaptor
        from latent_mechanics.online.loop import init_strategies

        model, table, _, _ = load_checkpoint(checkpoint, device=device,
                                             with_embeddings=True,
                                             stage="live_viewer")
        model.freeze()
        self.model = model

        if table is None:
            raise ValueError(
                f"{checkpoint} carries no embedding table, so there is no prior to "
                "initialise the latent from. Pass a stage-1/stage-4 training "
                "checkpoint (best.pt / last.pt), not an embeddings-stripped one.")
        train_latents = table.weight.detach().cpu().numpy()
        init = init_strategies(train_latents, seed)[init_name]

        self.kind = self._resolve_kind(kind)
        self.basis = None
        if self.kind == "ukf":
            from latent_mechanics.belief.adaptor import UKFLatentAdaptor
            self.basis = self._load_basis(checkpoint)
            self.adaptor = UKFLatentAdaptor(
                model, self.basis, init=init, prior_latents=train_latents,
                device=device)
        else:
            self.adaptor = GradientLatentAdaptor(
                model, init=init, lr=lr, device=device)

        self.init_name = init_name
        self._sq_err = np.zeros(2)
        self._n = 0
        self.n_skipped = 0

    # -- backend selection -------------------------------------------------
    @staticmethod
    def _available() -> bool:
        import importlib.util
        return importlib.util.find_spec("latent_mechanics.belief.adaptor") is not None

    @classmethod
    def _resolve_kind(cls, kind: str) -> str:
        if kind == "auto":
            return "ukf" if cls._available() else "gd"
        if kind == "ukf" and not cls._available():
            raise SystemExit(
                "--belief ukf was requested but latent_mechanics.belief is not on "
                "this branch. Use --belief gd (gradient-descent baseline), or run "
                "from a branch where the UKF work is merged.")
        return kind

    @staticmethod
    def _load_basis(checkpoint: str | Path):
        """The reduced basis must come from the same table as the predictor.

        ``load_or_create`` refuses to reuse a persisted basis fit on a different
        table, which is the behaviour we want; for any other checkpoint the basis
        is computed in memory rather than written to disk, because this tool is
        read-only.
        """
        from latent_mechanics.belief import basis as basis_mod

        if str(checkpoint) == basis_mod.DEFAULT_TABLE:
            return basis_mod.load_or_create(table_ckpt=str(checkpoint))
        return basis_mod.compute_basis(str(checkpoint))

    # -- driving -----------------------------------------------------------
    def observe(self, state: np.ndarray, action: float, next_state: np.ndarray) -> None:
        step = self.adaptor.observe(state, np.array([action], dtype=np.float32),
                                    next_state)
        self._sq_err += np.asarray(step.error, dtype=np.float64) ** 2
        self._n += 1

    def skip(self) -> None:
        """A transition the training set would have excluded (near a joint limit)."""
        self.n_skipped += 1

    def readout(self) -> BeliefReadout:
        b = self.adaptor.belief()
        mean = np.asarray(b["mean"], dtype=np.float64)
        cov = b.get("cov_reduced")
        if cov is None:
            cov = b.get("cov")
        n = max(self._n, 1)
        rmse = np.sqrt(self._sq_err / n)
        return BeliefReadout(
            z_norm=float(np.linalg.norm(mean)),
            z_reduced=(self.basis.encode(mean).reshape(-1)[:3]
                       if self.basis is not None else None),
            p_trace=(float(np.trace(np.atleast_2d(cov))) if cov is not None else None),
            rmse_q=float(rmse[0]),
            rmse_v=float(rmse[1]),
            n_updates=self._n,
            n_skipped=self.n_skipped,
        )

    def describe(self) -> str:
        bits = [f"backend={self.adaptor.name}", f"init={self.init_name}"]
        if self.basis is not None:
            bits.append(f"basis_dim={self.basis.dim}")
        return "  belief: " + "  ".join(bits)


# ---------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------

class LiveSession:
    """One mechanism, driven by a stream of the project's excitation episodes."""

    def __init__(self, args: argparse.Namespace, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.args = args
        self.frame_skip = int(args.frame_skip or cfg.sim.frame_skip)
        self.episode_seconds = float(args.episode_seconds or cfg.sim.episode_seconds)
        self.steps_per_episode = int(round(self.episode_seconds / dyn.DT))

        rng = np.random.default_rng(args.seed)
        # Draw ``instance`` mechanisms and keep the last, so --instance indexes
        # the same population order sample_population would produce.
        for i in range(args.instance + 1):
            self.params = lib.sample_params(args.family, rng, i)

        with in_repo_root():
            self.model = lib.build_model(self.params)
        self.data = mujoco.MjData(self.model)
        self.qadr, self.dof, self.jid = lib.joint_info(self.model)
        self.perturbations = lib.perturbations_for(self.params)
        self.lo = float(self.model.jnt_range[self.jid][0])
        self.hi = float(self.model.jnt_range[self.jid][1])
        self.pos_u, self.vel_u, self.force_u = units_for(self.params.joint_type)

        # Excitation config, with an optional forced profile kind.
        self.exc_cfg = cfg.excitation
        if args.profile:
            import copy
            self.exc_cfg = copy.deepcopy(cfg.excitation)
            self.exc_cfg.profile_weights = {args.profile: 1.0}

        self.belief: BeliefRunner | None = None
        if args.checkpoint:
            self.belief = BeliefRunner(
                args.checkpoint, kind=args.belief, init_name=args.init,
                lr=args.lr, device=args.device, seed=args.seed)

        # Live state
        self.episode = -1
        self.step_in_episode = 0
        self.sim_time = 0.0
        self.tau = 0.0
        self.tau_extra = 0.0
        self._tau_block: list[float] = []
        self._prev_sample: np.ndarray | None = None
        self.profile = None
        self.tau_fn: Callable[[float], float] = lambda t: 0.0
        self._start_episode(0)

    # -- episodes ----------------------------------------------------------
    def _start_episode(self, ep: int) -> None:
        """Draw a fresh excitation profile, exactly as data generation does.

        The seed formula is copied from ``rollout.rollout_mechanism`` so a given
        (seed, instance, episode) reproduces the torque stream the dataset used.
        """
        self.episode = ep
        self.step_in_episode = 0
        self._tau_block.clear()
        self._prev_sample = None  # never bridge a transition across a reset

        rng = np.random.default_rng(
            self.args.seed * 7717 + (self.params.mechanism_id + 1) * 1009 + ep)
        self.profile = lib.scaled_profile(
            self.exc_cfg, rng, self.steps_per_episode, self.frame_skip, self.params)
        self.tau_fn = self.profile.as_fn()

        if not self.args.no_reset:
            # Every training episode starts from the closed mechanism; see the
            # excitation module docstring for why the start state cannot simply
            # be randomised instead.
            mujoco.mj_resetData(self.model, self.data)
            for p in self.perturbations:
                p.reset(self.model)

    # -- one integrator step ----------------------------------------------
    def step(self) -> None:
        t_local = self.step_in_episode * dyn.DT
        q = float(self.data.qpos[self.qadr])
        v = float(self.data.qvel[self.dof])
        tau = float(self.tau_fn(t_local))

        extra = 0.0
        for p in self.perturbations:
            p.update_model(t_local, self.model)
            extra += p.extra_torque(t_local, q, v)

        # The one line this whole tool exists to drive: generalised force
        # straight onto the observed DOF. Same as rollout.simulate_mechanism.
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[self.dof] = tau + extra
        mujoco.mj_step(self.model, self.data)

        self.tau = tau              # the ACTION excludes unmodelled physics
        self.tau_extra = extra
        self._tau_block.append(tau)
        self.step_in_episode += 1
        self.sim_time += dyn.DT

        if self.step_in_episode % self.frame_skip == 0:
            self._on_model_boundary()
        if self.step_in_episode >= self.steps_per_episode:
            self._start_episode(self.episode + 1)

    def _on_model_boundary(self) -> None:
        """A model-rate sample has landed. Feed the belief update, if enabled.

        The transition convention matches ``data_gen.transitions_from_log``: the
        state recorded here is the state *after* the block of ``frame_skip``
        steps, and the action is the zero-order-hold torque applied across that
        block. The first boundary of an episode only seeds ``_prev_sample`` --
        there is no completed transition yet, which is exactly why the offline
        slicing starts at ``j = frame_skip - 1``.
        """
        sample = np.array([float(self.data.qpos[self.qadr]),
                           float(self.data.qvel[self.dof])], dtype=np.float32)
        block = np.asarray(self._tau_block, dtype=np.float64)
        self._tau_block.clear()

        prev, self._prev_sample = self._prev_sample, sample
        if prev is None or self.belief is None or block.size == 0:
            return

        action = float(block.mean())
        # The block is a zero-order hold by construction, so this is a cheap
        # assertion that the profile grid and the step counter stayed aligned.
        if float(np.max(np.abs(block - action))) > 1e-9 * max(abs(action), 1.0):
            raise RuntimeError(
                f"torque was not constant across a model timestep (spread "
                f"{np.ptp(block):.3e}); excitation grid and step counter have "
                "drifted apart")

        if near_limit_mask(prev[None, :], sample[None, :], self.lo, self.hi)[0]:
            # Training excluded these (SimConfig.exclude_near_limit): the
            # constraint torque at a stop is not part of the action, so the
            # transition is close to unpredictable from (s, a, z) alone.
            self.belief.skip()
            return
        self.belief.observe(prev, action, sample)

    # -- readout -----------------------------------------------------------
    def header(self) -> str:
        lines = [
            f"  {self.params.summary()}",
            f"  joint={self.params.joint_type} "
            f"range=[{self.lo:.3f}, {self.hi:.3f}] {self.pos_u}  "
            f"units: {self.pos_u} / {self.vel_u} / {self.force_u}",
            f"  dt={dyn.DT} s ({1 / dyn.DT:.0f} Hz)  frame_skip={self.frame_skip}  "
            f"dt_model={dyn.DT * self.frame_skip} s "
            f"({1 / (dyn.DT * self.frame_skip):.0f} Hz)",
            f"  episode={self.episode_seconds} s  "
            f"reset_between_episodes={not self.args.no_reset}",
        ]
        if self.belief is not None:
            lines.append(self.belief.describe())
        return "\n".join(lines)

    def readout_fields(self) -> list[tuple[str, str]]:
        q = float(self.data.qpos[self.qadr])
        v = float(self.data.qvel[self.dof])
        kind = self.profile.kind if self.profile else "-"
        fields = [
            ("t", f"{self.sim_time:7.2f} s"),
            ("episode", f"{self.episode:d} ({kind})"),
            ("q", f"{q:+8.4f} {self.pos_u}"),
            ("qdot", f"{v:+8.4f} {self.vel_u}"),
            ("tau", f"{self.tau:+8.3f} {self.force_u}"),
        ]
        if self.perturbations:
            fields.append(("tau_extra", f"{self.tau_extra:+8.3f} {self.force_u}"))
        if self.belief is not None:
            r = self.belief.readout()
            fields.append(("|z|", f"{r.z_norm:7.3f}"))
            if r.z_reduced is not None:
                fields.append(("z_pca",
                               " ".join(f"{c:+.2f}" for c in r.z_reduced)))
            fields.append(("tr(P)",
                           "n/a" if r.p_trace is None else f"{r.p_trace:9.3e}"))
            fields.append(("1-step rmse",
                           f"q {r.rmse_q:.3e} / v {r.rmse_v:.3e}"))
            fields.append(("updates", f"{r.n_updates:d} (+{r.n_skipped} skipped)"))
        return fields

    def console_line(self) -> str:
        return "  ".join(f"{k} {v}" for k, v in self.readout_fields())

    def overlay_columns(self) -> tuple[str, str]:
        fields = self.readout_fields()
        return ("\n".join(k for k, _ in fields), "\n".join(v for _, v in fields))


# ---------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------

def _pump(session: LiveSession, viewer: Any, args: argparse.Namespace) -> None:
    """Step the sim in wall-clock time, syncing and printing on their own clocks.

    The simulation catches up to where wall-clock says it should be, rather than
    sleeping a fixed amount per step: a 500 Hz Python loop cannot rely on
    millisecond sleep granularity, and a catch-up loop self-corrects instead of
    drifting slowly behind. The per-frame step cap stops a machine that cannot
    keep up from spiralling.
    """
    frame_dt = 1.0 / args.fps
    quiet = args.readout_hz <= 0
    readout_dt = 1.0 / args.readout_hz if not quiet else float("inf")
    max_steps_per_frame = max(1, int(4 * args.speed * frame_dt / dyn.DT) + 10)

    t0 = time.perf_counter()
    next_readout = 0.0
    overlay_ok = args.overlay and viewer is not None
    width = 0

    try:
        while viewer is None or viewer.is_running():
            if args.duration > 0 and session.sim_time >= args.duration:
                break

            target = (time.perf_counter() - t0) * args.speed
            n = 0
            while session.sim_time < target and n < max_steps_per_frame:
                session.step()
                n += 1
                if args.duration > 0 and session.sim_time >= args.duration:
                    break

            if viewer is not None:
                viewer.sync()

            now = time.perf_counter() - t0
            if not quiet and now >= next_readout:
                next_readout = now + readout_dt
                line = session.console_line()
                width = max(width, len(line))
                print("\r" + line.ljust(width), end="", flush=True)
                if overlay_ok:
                    try:
                        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,
                                          mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                          *session.overlay_columns()))
                    except Exception as exc:  # bonus feature; never fatal
                        overlay_ok = False
                        print(f"\n  (on-screen overlay disabled: {exc})")

            slack = frame_dt - (time.perf_counter() - t0 - now)
            if slack > 0:
                time.sleep(slack)
    except KeyboardInterrupt:
        pass
    finally:
        print()


def run(args: argparse.Namespace, cfg: ExperimentConfig) -> None:
    session = LiveSession(args, cfg)
    print(f"\nlive_viewer  family={args.family}  seed={args.seed}  "
          f"instance={args.instance}")
    print(session.header())
    print()

    if args.no_viewer:
        _pump(session, None, args)
    else:
        import mujoco.viewer
        try:
            ctx = mujoco.viewer.launch_passive(
                session.model, session.data,
                show_left_ui=False, show_right_ui=False)
        except RuntimeError as exc:
            if sys.platform == "darwin":
                raise SystemExit(
                    f"{exc}\n\nOn macOS the passive viewer must be launched with "
                    "mjpython, not python:\n"
                    f"    mjpython {' '.join(sys.argv)}\n"
                    "Or pass --no-viewer to run the physics and readout without a "
                    "window.") from exc
            raise
        with ctx as viewer:
            _pump(session, viewer, args)

    if session.belief is not None:
        r = session.belief.readout()
        print(f"  final: {r.n_updates} belief updates at "
              f"{1 / (dyn.DT * session.frame_skip):.0f} Hz "
              f"({r.n_skipped} skipped near a joint limit)")
        print(f"         1-step rmse  q {r.rmse_q:.4e} {session.pos_u}  "
              f"v {r.rmse_v:.4e} {session.vel_u}")
        # The predictor is frozen; this is the check that says so.
        session.belief.adaptor.assert_network_unchanged()
        print("         predictor weights unchanged (verified)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", default="door", choices=list(lib.FAMILIES),
                   help="mechanism family to build (default: door)")
    p.add_argument("--instance", type=int, default=0,
                   help="which sampled instance of that family (default: 0)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for mechanism sampling and excitation (default: 0)")
    p.add_argument("--duration", type=float, default=0.0,
                   help="simulated seconds to run; 0 runs until interrupted")
    p.add_argument("--speed", type=float, default=1.0,
                   help="real-time multiplier: 0.5 = half speed (default: 1.0)")
    p.add_argument("--profile", default=None,
                   choices=["multisine", "steps", "chirp", "swing"],
                   help="force one excitation profile instead of sampling the mix")
    p.add_argument("--episode-seconds", type=float, default=None,
                   help="override the excitation episode length")
    p.add_argument("--frame-skip", type=int, default=None,
                   help="override frame_skip (belief-update cadence)")
    p.add_argument("--no-reset", action="store_true",
                   help="do not reset to the closed state between episodes")
    p.add_argument("--config", default=None,
                   help="YAML config to take excitation/sim defaults from")

    g = p.add_argument_group("readout")
    g.add_argument("--readout-hz", type=float, default=5.0,
                   help="console refresh rate; 0 disables (default: 5)")
    g.add_argument("--fps", type=float, default=60.0,
                   help="viewer sync rate (default: 60)")
    g.add_argument("--overlay", action=argparse.BooleanOptionalAction, default=True,
                   help="also draw the readout in the viewer window")
    g.add_argument("--no-viewer", action="store_true",
                   help="run physics and readout without opening a window")

    g = p.add_argument_group("belief pipeline (step 3)")
    g.add_argument("--checkpoint", default=None,
                   help="trained predictor; enables the live belief update")
    g.add_argument("--belief", default="auto", choices=["auto", "gd", "ukf"],
                   help="belief backend (default: auto -- ukf if available, else gd)")
    g.add_argument("--init", default="zero",
                   choices=["zero", "mean", "medoid", "random_trained"],
                   help="starting latent (default: zero)")
    g.add_argument("--lr", type=float, default=0.03,
                   help="latent learning rate, gradient backend only")
    g.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.speed <= 0:
        raise SystemExit("--speed must be positive")
    cfg = load_config(args.config)
    run(args, cfg)


if __name__ == "__main__":
    main()
