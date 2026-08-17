"""Latent mechanics embeddings for articulated-object dynamics, alongside the RLS
baseline in ``run_door_dynamics_validation.py`` (never modified here).

Stage 1 learns ``(state, action, z) -> next_state`` with one embedding row per
training door. Stage 2 keeps the network, drops the table, and optimises a fresh
``z`` online for an unseen door.
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
