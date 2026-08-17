"""Stage 2: online latent adaptation on an unseen door. The Stage-1 network stays
frozen and only the latent changes, one transition at a time.

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
