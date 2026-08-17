"""Experiments. One directory per question.

Each directory holds a ``run.py`` containing an ``ExperimentSpec`` and nothing
else of substance. Reading the spec should tell you the plant, the observation
model, the predictor, the mechanics representation, the estimators, the
initialisation, the disturbances and the metrics -- and therefore exactly what
hypothesis is being tested.

    sanity_one_door/    the smallest end-to-end run; start here
    estimator_convergence/  does adaptation beat the control, and does RLS win?
    observation_noise/  the sensing-crossover claim, on a chosen predictor

Run one with:  python3.10 -m experiments.<name>.run
"""

from experiments._spec import DEFAULT_METHODS, ExperimentSpec, run_experiment

__all__ = ["ExperimentSpec", "run_experiment", "DEFAULT_METHODS"]
