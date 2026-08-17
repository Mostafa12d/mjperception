## Agent Prompt: Research Codebase Audit and Refactor

You are taking over an active research codebase for a project on **online mechanics estimation for generalized articulated object manipulation**.

Before implementing anything new, I need you to deeply understand the existing codebase and help reorganize it for continued research.

### 1. First, explain the existing project before changing anything

Do not modify code initially.

Inspect the entire relevant codebase and produce a clear technical report answering:

#### A. What is the research goal?

Explain, in plain language and mathematically where useful:

* What problem are we trying to solve?
* What is the central hypothesis?
* What does the learned dynamics predictor do?
* What is the mechanics belief?
* What is estimated online?
* What role does the UKF currently play?
* What are the observations?
* What is considered an action/control input?
* What is predicted?
* What is compared against reality to generate the innovation?
* What experiments have been implemented so far?
* What has actually been demonstrated versus what is still only a hypothesis?

Be critical. Do not describe the current implementation as if it is necessarily the correct final solution.

#### B. Trace the actual data flow

For the current implementation, trace one complete timestep:

[
\text{simulation / observations}
\rightarrow
\text{predictor input}
\rightarrow
\text{predicted next state}
\rightarrow
\text{observed next state}
\rightarrow
\text{residual}
\rightarrow
\text{online estimator}
\rightarrow
\text{updated mechanics belief}.
]

For every arrow, identify:

* which variable is passed,
* where it is produced,
* where it is consumed,
* whether it is simulator ground truth, synthetic observation, learned output, or estimated state.

I want the actual code path, not the intended architecture.

#### C. Identify architectural problems

Identify places where the code has become:

* overly coupled,
* difficult to test,
* difficult to replace components,
* dependent on assumptions hidden in unrelated modules,
* prematurely designed around one particular estimator or predictor,
* duplicating experiment logic,
* mixing simulation, learning, estimation, and evaluation.

Also identify dead abstractions or abstractions that add complexity without helping research.

---

# 2. Define the research architecture before refactoring

Based on the actual codebase and the research goal, propose a **minimal experimental architecture**.

The goal is not maximum modularity.

The goal is:

> I should be able to replace the predictor, observation model, mechanics representation, or online estimator without rewriting the rest of the experiment.

At a high level, I currently believe the research loop should look approximately like:

[
\text{Environment}
\rightarrow
\text{Observation}
\rightarrow
\text{Dynamics Predictor}
\rightarrow
\text{Predicted Observation}
]

[
\text{Predicted Observation}
----------------------------

\text{Actual Observation}
\rightarrow
\text{Innovation}
\rightarrow
\text{Online Mechanics Estimator}
\rightarrow
\text{Updated Mechanics Belief}.
]

However, do not blindly enforce this architecture. Check whether it is mathematically and architecturally correct for the current project.

Propose the smallest clean set of interfaces required.

For example, something conceptually like:

```text
Environment / Plant
        │
        ▼
Observation Model
        │
        ▼
Observation
        │
        ├───────────────┐
        ▼               │
Dynamics Predictor      │
        │               │
Predicted Observation   │
        │               │
        ▼               ▼
       Innovation
            │
            ▼
    Online Estimator
            │
            ▼
     Mechanics Belief
```

The exact interfaces should emerge from the audit.

---

# 3. Preserve experimentation flexibility

This is critical.

Do **not** design the code as if:

* the UKF is definitely the final estimator,
* the latent embedding is definitely the final mechanics representation,
* the neural predictor is definitely the final predictor,
* the current observation vector is correct,
* the current simulator is the final environment.

The codebase should support questions such as:

### Predictor experiments

Can I easily test:

* learned neural dynamics,
* analytical dynamics,
* intentionally misspecified dynamics,
* different network architectures?

### Mechanics representation experiments

Can I test:

* latent mechanics vectors,
* explicit physical parameters,
* hybrid representations?

### Online estimation experiments

Can I replace:

```text
UKF
```

with:

```text
Gradient Descent
RLS
EKF
Particle Filter
GRU-based learned update
Other estimators
```

without rewriting the experiment driver?

### Observation experiments

Can I move from:

```text
ground-truth joint position / velocity
```

to:

```text
noisy joint estimates
vision-based estimates
robot pose
robot velocity
force/torque
partial observations
```

without rewriting the estimator itself?

Do not over-engineer this. Use simple interfaces and dependency injection where useful.

---

# 4. Refactor in stages

Do not perform a giant destructive rewrite immediately.

Instead:

### Phase 1: Audit

Produce:

1. `CURRENT_SYSTEM.md`

containing:

* research objective,
* current architecture,
* actual data flow,
* module responsibilities,
* implemented experiments,
* known limitations,
* confusing or redundant components.

2. `REFACTOR_PROPOSAL.md`

containing:

* proposed architecture,
* rationale,
* old-to-new module mapping,
* components to preserve,
* components to simplify,
* components to remove,
* migration plan.

Do not modify functioning code during this phase.

---

### Phase 2: Build the minimal experimental core

After the audit, refactor incrementally.

The core experiment should make the following loop obvious:

```python
belief = estimator.initialize(...)

for observation, action, next_observation in trajectory:

    prediction = predictor.predict(
        observation,
        action,
        belief,
    )

    belief = estimator.update(
        belief=belief,
        observation=observation,
        action=action,
        prediction=prediction,
        next_observation=next_observation,
    )
```

This is conceptual pseudocode. Do not force this exact API if a better minimal design emerges.

The important requirement is that a researcher can quickly answer:

* What is observed?
* What is predicted?
* What generates the residual?
* What state is being estimated?
* What updates the estimate?

---

### Phase 3: Preserve all existing results

The refactor must not silently change experimental behavior.

For every migrated experiment:

1. Run the old implementation.
2. Record its metrics.
3. Run the refactored implementation with equivalent configuration.
4. Compare outputs within an appropriate tolerance.

Existing working baselines, especially RLS, should remain intact.

Do not "improve" algorithms while refactoring.

Separate:

```text
refactoring changes
```

from

```text
algorithmic changes.
```

---

# 5. Create a research-friendly experiment system

I want to be able to add experiments without creating another large package.

Experiments should answer explicit questions.

For example:

```text
experiments/
    predictor_accuracy/
    estimator_convergence/
    transient_disturbance/
    model_mismatch/
    observation_noise/
    cross_mechanism_generalization/
```

Each experiment should clearly specify:

* plant/environment,
* observation model,
* predictor,
* mechanics representation,
* estimator,
* initialization,
* disturbances,
* evaluation metrics.

I should be able to look at an experiment configuration and understand exactly what hypothesis is being tested.

---

# 6. Add a system-level sanity experiment

Before adding new research ideas, create one small canonical experiment that is easy to understand.

For example:

1. Choose one articulated mechanism.
2. Generate an interaction trajectory.
3. Start with an incorrect mechanics belief.
4. Predict the next observation.
5. Compare prediction with the actual observation.
6. Update the belief.
7. Visualize:

[
\text{true behavior}
]

versus

[
\text{prediction}
]

and

[
\text{mechanics belief over time}.
]

The goal is not performance.

The goal is to create the smallest experiment where the entire system can be understood and debugged end-to-end.

---

# 7. Final deliverable

At the end, provide:

### A. A research-level explanation

Explain the final architecture in enough detail that I can understand:

> What are we trying to prove, and what exactly happens at every timestep?

### B. A code map

For every major module:

* what it does,
* what it receives,
* what it outputs,
* what research component it corresponds to.

### C. A "how to modify this project" guide

Show concrete examples:

> If I want to test a new observation...

what files do I change?

> If I want to replace the UKF...

what interface do I implement?

> If I want to test a different predictor...

what changes?

> If I want to add a transient disturbance experiment...

where does it go?

### D. Do not continue implementing new algorithms

Stop after the refactor and validation.

The next algorithmic decision should be made after I understand the simplified system.

---

## Guiding principle

The project is still exploratory research.

**Do not optimize for the final architecture. Optimize for the ability to understand, test, break, replace, and improve ideas quickly.**

If an abstraction makes experimentation harder, remove it.

If two modules can simply be functions, do not create a framework around them.

At every major design decision, ask:

> "Does this make the research hypothesis easier to test?"

If not, do not add it.
