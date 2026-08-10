"""
Stage 2: online latent adaptation.

A robot meets an unseen door and improves its mechanics belief continuously
while interacting with it. The Stage-1 dynamics network stays completely frozen;
only the latent changes.

    z_0  --(observe one transition)-->  z_1  -->  z_2  -->  ...  -->  z_T

Stage 1 is imported, never modified.

Entry points:
    python3.10 -m latent_mechanics.online.experiments
    python3.10 -m latent_mechanics.online.tests
"""

from latent_mechanics.online.adaptor import (
    AdaptorStep,
    GradientLatentAdaptor,
    OnlineAdaptor,
    OnlineLatentAdaptor,
)
from latent_mechanics.online.config import OnlineConfig, load_config
from latent_mechanics.online.loop import (
    AdaptationLog,
    episode_stream,
    init_strategies,
    run_online_adaptation,
)
from latent_mechanics.online.rls_adaptor import RLSAdaptor

__all__ = [
    "OnlineAdaptor",
    "OnlineLatentAdaptor",
    "GradientLatentAdaptor",
    "RLSAdaptor",
    "AdaptorStep",
    "AdaptationLog",
    "run_online_adaptation",
    "episode_stream",
    "init_strategies",
    "OnlineConfig",
    "load_config",
]
