# mjperception

MuJoCo-based research on **estimating the mechanics of articulated objects while
interacting with them** — how heavy a door is, how much friction its hinge has,
whether it has a closer spring — and on doing that estimation *online*, from the
same interaction the robot is already performing.

Two approaches are developed side by side and benchmarked against each other:
classical recursive least squares on a linear-in-parameters model, and a learned
dynamics network carrying a per-object latent "mechanics vector".

- **[docs/RUNNING.md](docs/RUNNING.md)** — how to set up and run everything.
- **[docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)** — what each stage established.

---

## Layout

| directory | line of work |
|---|---|
| [baseline/](baseline/) | **A** — RLS system identification + adaptive impedance on the bare door. `run_door_dynamics_validation.py` is the spine: its `simulate()` log dict is the data contract every other estimator consumes. |
| [iiwa/](iiwa/) | **B** — the same loop driven through a KUKA iiwa 14, with proprioceptive FK and a simulated wrist F/T sensor instead of oracle state. |
| [latent_mechanics/](latent_mechanics/) | **C** — the learned alternative. Five evaluation stages plus `geometry/` and `belief/`; each subpackage has its own README with results and limitations. |
| [perception/](perception/) | **D** — RGB-D capture, FlowBot3D articulation flow, OMIP kinematic-structure estimation. |
| [scenes/](scenes/) | every MuJoCo XML and mesh, plus the `scene_path()` resolver that anchors them to the filesystem rather than the working directory. |
| [tools/](tools/) | `live_viewer.py` — watch a mechanism move under the project's own excitation. |
| [configs/](configs/) | YAML for the latent-mechanics stages. |
| `data/` `runs/` `media/` | generated artifacts, all gitignored and reproducible. |
| [archive/](archive/) | `park/` (parked experiments) and `notes/` (handover documents). |

## Running anything

Commands are invoked as modules from the repository root:

```bash
python3.10 -m baseline.run_door_dynamics_validation
python3.10 -m latent_mechanics.online.experiments --config configs/online_adaptation.yaml
```

Setup and the full per-script reference live in [docs/RUNNING.md](docs/RUNNING.md).

## Tests

Seven self-check suites, each exiting non-zero on failure:

```bash
for t in latent_mechanics.tests latent_mechanics.online.tests \
         latent_mechanics.mismatch.tests latent_mechanics.mechanisms.tests \
         latent_mechanics.curriculum.tests latent_mechanics.geometry.tests \
         latent_mechanics.belief.test_ukf_reference; do python3.10 -m $t; done
```

They cover more than shapes and types: the frozen-network contract in Stage 2,
episode-level split disjointness, and a bit-exact agreement check between the
Stage-3 perturbed integrator and the Stage-1 one. Run them first when a result
looks surprising.
