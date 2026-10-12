"""Role 3: training loop.

train(model, dataset, cfg) -> TrainResult(checkpoint_path, history)
Requirements: mini-batch coordinate sampling, seeds, logging, checkpointing,
reproducible from a config + git commit.
"""
from dataclasses import dataclass, field


# 训练结果同时保存检查点路径和训练历史。
@dataclass
class TrainResult:
    checkpoint_path: str
    history: dict = field(default_factory=dict)


def train(model, dataset, cfg) -> TrainResult:
    """TODO(Role 3)."""
    raise NotImplementedError
