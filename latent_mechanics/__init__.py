"""
Latent mechanics embeddings for articulated-object dynamics.

Research direction that sits *alongside* the RLS system-identification baseline
in ``run_door_dynamics_validation.py`` (which this package never modifies).

Stage 1 (this package): learn a single dynamics model

    (state, action, z) -> next_state

where ``z`` is a learned latent vector describing the hidden mechanics of one
door. During stage 1 every training door owns one row of an embedding table.

Stage 2 (future): keep the same network, throw the table away, and optimise a
fresh ``z`` online for an unseen door. Nothing in ``MechanicsDynamicsModel``
knows that ``z`` ever came from a table -- see ``model.py``.
"""

from latent_mechanics.config import ExperimentConfig, load_config
from latent_mechanics.model import (
    DoorEmbeddingTable,
    MechanicsDynamicsModel,
    load_checkpoint,
    save_checkpoint,
)

__all__ = [
    "ExperimentConfig",
    "load_config",
    "MechanicsDynamicsModel",
    "DoorEmbeddingTable",
    "save_checkpoint",
    "load_checkpoint",
]
