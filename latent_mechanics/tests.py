"""
Self-checks for the latent-mechanics stage-1 pipeline.

These pin the properties that are easy to break silently later, above all the
stage-2 contract: the dynamics model must accept an arbitrary latent tensor and
must let gradients flow into it while every network weight stays frozen. If that
test fails, online embedding optimisation is impossible no matter how good the
stage-1 numbers look.

Run:
    python3.10 -m latent_mechanics.tests
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

import run_door_dynamics_validation as dyn
from latent_mechanics import door_sampler
from latent_mechanics.config import ExperimentConfig, load_config
from latent_mechanics.data_gen import episode_length, generate_dataset, save_dataset
from latent_mechanics.dataset import DoorTransitionDataset
from latent_mechanics.excitation import sample_profile
from latent_mechanics.model import (
    DoorEmbeddingTable,
    MechanicsDynamicsModel,
    load_checkpoint,
    save_checkpoint,
)
from latent_mechanics.rollout import multistart_rollout, rollout

_FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)


# ---------------------------------------------------------------------------

def test_rls_baseline_untouched() -> None:
    """The baseline must keep working exactly as before, with its globals intact."""
    print("\nRLS baseline integrity")
    before = (dyn.T_END, dyn.N_STEPS, dyn.DT)
    with episode_length(2.0):
        inner = (dyn.T_END, dyn.N_STEPS)
    after = (dyn.T_END, dyn.N_STEPS, dyn.DT)
    check("episode_length restores dyn globals", before == after, f"{before} != {after}")
    check("episode_length actually applies", inner == (2.0, 1000), str(inner))

    state = dyn.rls_init(2, lam=0.99)
    state = dyn.rls_step(state, np.array([1.0, 0.5]), 2.0)
    check("dyn.rls_step still runs", np.all(np.isfinite(state.theta)))


def test_model_contract() -> None:
    """The dynamics model must not know where its latent came from."""
    print("\nMechanicsDynamicsModel interface")
    model = MechanicsDynamicsModel(embed_dim=8, hidden_sizes=[32, 32])
    s = torch.randn(16, 2)
    a = torch.randn(16, 1)

    out = model(s, a, torch.randn(16, 8))
    check("batched latent -> (B, 2)", out.shape == (16, 2), str(out.shape))

    shared = model(s, a, torch.randn(8))
    check("single latent broadcasts over batch", shared.shape == (16, 2), str(shared.shape))

    # A latent that never came from any table at all.
    free = torch.nn.Parameter(torch.zeros(1, 8))
    check("accepts a standalone nn.Parameter", model(s, a, free).shape == (16, 2))

    m_delta = MechanicsDynamicsModel(embed_dim=4, hidden_sizes=[16], predict_delta=True)
    with torch.no_grad():
        # Zero the output head so the predicted delta is exactly delta_mean.
        m_delta.net[-1].weight.zero_()
        m_delta.net[-1].bias.zero_()
    pred = m_delta(s, a, torch.zeros(16, 4))
    check(
        "predict_delta adds the delta to the input state",
        torch.allclose(pred, s + m_delta.delta_mean, atol=1e-6),
    )


def test_stage2_contract() -> None:
    """Freeze the network, optimise only z -- the stage-2 pipeline in miniature."""
    print("\nStage-2 contract (frozen network, optimisable latent)")
    model = MechanicsDynamicsModel(embed_dim=8, hidden_sizes=[32, 32]).freeze()
    check(
        "freeze() clears requires_grad on every weight",
        all(not p.requires_grad for p in model.parameters()),
    )

    z = model.new_latent(1)
    check("new_latent is an optimisable parameter", z.requires_grad and z.shape == (1, 8))
    check("new_latent defaults to the zero (average-door) prior", torch.all(z == 0))

    s, a = torch.randn(32, 2), torch.randn(32, 1)
    target = torch.randn(32, 2)
    opt = torch.optim.Adam([z], lr=1e-2)

    first = None
    for _ in range(50):
        loss = torch.nn.functional.mse_loss(model(s, a, z), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss)

    check("gradients reach the standalone latent", torch.any(z != 0))
    check("optimising only z reduces the loss", float(loss) < first, f"{float(loss)} vs {first}")
    check(
        "network weights did not move",
        all(p.grad is None for p in model.parameters()),
    )


def test_checkpoint_roundtrip(cfg: ExperimentConfig) -> None:
    print("\nCheckpoint round-trip")
    model = MechanicsDynamicsModel(embed_dim=8, hidden_sizes=[32, 32])
    with torch.no_grad():
        model.set_norm_stats(
            {
                "state_mean": torch.tensor([0.5, -0.1]),
                "state_std": torch.tensor([0.3, 0.2]),
                "action_mean": torch.tensor([1.0]),
                "action_std": torch.tensor([2.0]),
                "delta_mean": torch.tensor([0.01, 0.02]),
                "delta_std": torch.tensor([0.03, 0.04]),
            }
        )
    table = DoorEmbeddingTable(5, 8)
    cfg.model.embed_dim = 8
    cfg.model.hidden_sizes = [32, 32]

    s, a, z = torch.randn(4, 2), torch.randn(4, 1), torch.randn(4, 8)
    expected = model(s, a, z)

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ckpt.pt"
        save_checkpoint(p, model, table, cfg)
        m2, t2, _, _ = load_checkpoint(p)
        check("weights round-trip", torch.allclose(m2(s, a, z), expected, atol=1e-6))
        check("normalisation buffers round-trip",
              torch.allclose(m2.state_std, model.state_std))
        check("embedding table round-trips",
              t2 is not None and torch.allclose(t2.weight, table.weight))

        m3, t3, _, _ = load_checkpoint(p, with_embeddings=False)
        check("stage-2 load can skip the table", t3 is None)
        check("model still usable without the table",
              torch.allclose(m3(s, a, z), expected, atol=1e-6))


def test_data_pipeline(cfg: ExperimentConfig) -> None:
    """Generate a tiny dataset and verify its structure and physical fidelity."""
    print("\nData pipeline")
    small = load_config(None)
    small.doors.n_train_doors = 3
    small.doors.n_heldout_doors = 1
    small.sim.episodes_per_door = 3
    small.sim.val_episodes_per_door = 1
    small.sim.episode_seconds = 2.0
    pack = generate_dataset(small, verbose=False)

    n = len(pack["state"])
    check("state is [angle, velocity]", pack["state"].shape == (n, 2))
    check("action is a scalar torque", pack["action"].shape == (n, 1))
    check("next_state matches state shape", pack["next_state"].shape == (n, 2))
    check("every sample carries a door id", pack["door_id"].shape == (n,))

    # Consecutive transitions must chain: next_state[i] == state[i+1] inside an episode.
    ptr = pack["episode_ptr"]
    lo, hi = int(ptr[0]), int(ptr[1])
    chained = np.allclose(
        pack["next_state"][lo : hi - 1], pack["state"][lo + 1 : hi], atol=1e-6
    )
    check("transitions chain within an episode", chained)

    # Held-out doors must not occupy an embedding row.
    heldout_ids = pack["door_id"][pack["split"] == 2]
    check(
        "held-out door ids sit above the embedding table",
        heldout_ids.min() >= small.doors.n_train_doors,
    )

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "tiny.npz"
        save_dataset(pack, p)
        tr = DoorTransitionDataset(p, "train")
        va = DoorTransitionDataset(p, "val")
        check("train/val episodes are disjoint",
              not (set(tr.episode_ids()) & set(va.episode_ids())))
        check("every training door appears in train",
              len(tr.door_ids) == small.doors.n_train_doors)
        check("val covers the same doors as train",
              set(va.door_ids) == set(tr.door_ids))
        stats = tr.norm_stats()
        check("norm stats are finite and positive",
              all(torch.all(torch.isfinite(v)) for v in stats.values())
              and bool(stats["delta_std"].min() > 0))


def test_transition_fidelity() -> None:
    """A recorded transition must be what MuJoCo actually does.

    Re-simulates one door from rest with the same zero-order-hold torque and
    checks the logged ``(state, action, next_state)`` triples against a fresh
    run. This is what catches an off-by-one in the state/torque alignment.
    """
    print("\nTransition fidelity vs MuJoCo")
    cfg = load_config(None)
    params = door_sampler.sample_door_params(cfg.doors, np.random.default_rng(3), 0)
    model = door_sampler.build_model(params)
    profile = sample_profile(
        cfg.excitation, np.random.default_rng(4), 1000, 10, params.frictionloss
    )

    from latent_mechanics.data_gen import transitions_from_log

    with episode_length(2.0):
        log = dyn.simulate(profile.as_fn(), model=model)
    tr = transitions_from_log(log, 10)

    # The action must equal the commanded hold for that interval.
    tau_fn = profile.as_fn()
    k = 5
    j = 10 - 1 + 10 * k  # start index of the k-th transition
    commanded = tau_fn((j + 1) * dyn.DT)
    check(
        "recorded action equals the commanded hold",
        abs(float(tr["action"][k, 0]) - commanded) < 1e-3 * max(abs(commanded), 1.0),
        f"{float(tr['action'][k, 0]):.6f} vs {commanded:.6f}",
    )
    check(
        "recorded state matches the MuJoCo log at that index",
        abs(float(tr["state"][k, 0]) - log["theta"][j]) < 1e-12,
    )
    check(
        "recorded next_state is frame_skip steps later",
        abs(float(tr["next_state"][k, 0]) - log["theta"][j + 10]) < 1e-12,
    )


def test_rollout_shapes() -> None:
    print("\nRollout")
    model = MechanicsDynamicsModel(embed_dim=4, hidden_sizes=[16]).freeze()
    z = torch.zeros(4)
    traj = rollout(model, z, torch.zeros(2), torch.randn(20, 1))
    check("rollout returns horizon + 1 states", traj.shape == (21, 2), str(traj.shape))

    from latent_mechanics.dataset import Episode

    T = 30
    ep = Episode(
        episode_id=0, door_id=0, kind="test",
        state=np.zeros((T, 2), np.float32),
        action=np.zeros((T, 1), np.float32),
        next_state=np.zeros((T, 2), np.float32),
        t=np.arange(T, dtype=np.float32),
        near_limit=np.zeros(T, bool),
    )
    pred, truth = multistart_rollout(model, z, ep, horizon=5)
    check("multistart covers every start", len(pred) == T - 5 + 1, str(len(pred)))
    check("prediction and truth align", pred.shape == truth.shape)

    ep.near_limit[10:] = True
    pred2, _ = multistart_rollout(model, z, ep, horizon=5, exclude_near_limit=True)
    check("limit windows are excluded", len(pred2) == 6, str(len(pred2)))


def test_checkpoint_provenance(cfg: ExperimentConfig) -> None:
    """A5: loading a predictor must record its hash, and a pin must be enforced."""
    import tempfile

    from latent_mechanics import provenance
    from latent_mechanics.model import (
        DoorEmbeddingTable,
        build_model_from_config,
        load_checkpoint,
        save_checkpoint,
    )

    print("\nCheckpoint provenance is recorded and pinnable")
    stats = {
        "state_mean": torch.zeros(2), "state_std": torch.ones(2),
        "action_mean": torch.zeros(1), "action_std": torch.ones(1),
        "delta_mean": torch.zeros(2), "delta_std": torch.ones(2),
    }
    model = build_model_from_config(cfg.model, stats)
    table = DoorEmbeddingTable(num_doors=7, embed_dim=cfg.model.embed_dim)

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "best.pt"
        save_checkpoint(p, model, table, cfg)

        digest = provenance.file_sha256(p)
        check("sha256 is a 64-char hex digest",
              len(digest) == 64 and all(c in "0123456789abcdef" for c in digest))
        check("hashing is stable across calls", provenance.file_sha256(p) == digest)

        provenance.set_quiet(True)
        try:
            _, t, _, _ = load_checkpoint(p, stage="unit_test")
            check("load records the stage and hash",
                  provenance.loaded().get("unit_test", ("", ""))[0] == digest)
            check("recorded table_rows matches the saved table", t.num_doors == 7)

            # A correct pin, full and truncated, must pass.
            load_checkpoint(p, stage="unit_test", expected_sha256=digest)
            load_checkpoint(p, stage="unit_test", expected_sha256=digest[:16])
            check("a matching pin (full and 16-char prefix) is accepted", True)

            # A wrong pin must raise, not warn.
            raised = False
            try:
                load_checkpoint(p, stage="unit_test", expected_sha256="deadbeef")
            except ValueError:
                raised = True
            check("a mismatched pin raises ValueError", raised)

            # Changing the file changes the hash, so substitution is detectable.
            table2 = DoorEmbeddingTable(num_doors=9, embed_dim=cfg.model.embed_dim)
            save_checkpoint(p, model, table2, cfg)
            check("a different checkpoint at the same path hashes differently",
                  provenance.file_sha256(p) != digest)
            raised = False
            try:
                load_checkpoint(p, stage="unit_test", expected_sha256=digest)
            except ValueError:
                raised = True
            check("the old pin now rejects the substituted checkpoint", raised)
        finally:
            provenance.set_quiet(False)

    # Every stage in the provenance table must name a real source.
    for stage, source, pattern in provenance.STAGE_SOURCES:
        check(f"provenance table entry '{stage}' names its source",
              bool(stage and source and pattern))


def main() -> None:
    print("latent_mechanics self-checks")
    cfg = load_config(None)
    test_rls_baseline_untouched()
    test_model_contract()
    test_stage2_contract()
    test_checkpoint_roundtrip(cfg)
    test_data_pipeline(cfg)
    test_transition_fidelity()
    test_rollout_shapes()
    test_checkpoint_provenance(cfg)

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILED: {', '.join(_FAILURES)}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
