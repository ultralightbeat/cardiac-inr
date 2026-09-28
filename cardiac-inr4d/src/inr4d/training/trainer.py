"""Role 3: training loop.

train(model, dataset, cfg) -> TrainResult(checkpoint_path, history)
Requirements: mini-batch coordinate sampling, seeds, logging, checkpointing,
reproducible from a config + git commit.
"""
from dataclasses import dataclass, field


@dataclass
class TrainResult:
    checkpoint_path: str
    history: dict = field(default_factory=dict)


def train(model, dataset, cfg) -> TrainResult:
    """TODO(Role 3)."""
    raise NotImplementedError
