"""Training loop: Adam on random mini-batches of (coordinate, intensity) pairs, MSE loss.

train(model, data, cfg) works with anything that has .sample(n) -> (coords, values)
(CoordinateDataset or TensorSampler below). The network is deliberately overfitted
to one patient: this is the standard setting for neural fields.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class TrainConfig:
    steps: int = 3000
    batch_size: int = 16384
    lr: float = 1e-4
    lr_final: float = 1e-5        # cosine decay from lr to lr_final
    log_every: int = 100
    seed: int = 0
    threads: int | None = None    # torch CPU threads; None = torch default
    device: str = "cpu"           # "cpu", "cuda", "xpu" or "auto" (see resolve_device)


@dataclass
class TrainResult:
    checkpoint_path: str | None
    history: dict = field(default_factory=dict)
    seconds: float = 0.0


def resolve_device(name="auto") -> str:
    """'auto' -> 'cuda' (NVIDIA, e.g. Colab T4) if available, else 'xpu' (Intel GPU) if available,
    else 'cpu'. Any other value is returned unchanged."""
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


class TensorSampler:
    """Random mini-batches from in-memory arrays."""

    def __init__(self, coords, values, seed=0):
        self.coords = torch.as_tensor(np.asarray(coords), dtype=torch.float32)
        self.values = torch.as_tensor(np.asarray(values), dtype=torch.float32).reshape(-1, 1)
        self.gen = torch.Generator().manual_seed(seed)

    def __len__(self):
        return len(self.coords)

    def sample(self, n):
        idx = torch.randint(len(self.coords), (n,), generator=self.gen)
        return self.coords[idx], self.values[idx]


def train(model, data, cfg: TrainConfig, checkpoint_path=None, eval_fn=None,
          eval_every=None, verbose=True) -> TrainResult:
    """Fit model to data. eval_fn(model) -> dict is called every eval_every steps."""
    if cfg.threads:
        torch.set_num_threads(int(cfg.threads))
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    hist = {"step": [], "loss": [], "train_psnr": [], "eval": []}
    t0 = time.time()
    run = 0.0
    model.train()
    for step in range(1, cfg.steps + 1):
        frac = (step - 1) / max(cfg.steps - 1, 1)
        lr = cfg.lr_final + 0.5 * (cfg.lr - cfg.lr_final) * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = data.sample(cfg.batch_size)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        loss = torch.mean((model(x) - y) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        run = 0.9 * run + 0.1 * loss.item() if step > 1 else loss.item()
        if step % cfg.log_every == 0 or step == cfg.steps:
            hist["step"].append(step)
            hist["loss"].append(run)
            hist["train_psnr"].append(10 * math.log10(4.0 / max(run, 1e-12)))
            if verbose:
                el = time.time() - t0
                eta = el / step * (cfg.steps - step)
                print(f"  step {step:5d}/{cfg.steps}  loss {run:.2e}  "
                      f"~train PSNR {hist['train_psnr'][-1]:5.2f} dB  "
                      f"{el:6.0f}s elapsed, ~{eta:5.0f}s left", flush=True)
        if eval_fn is not None and eval_every and (step % eval_every == 0 or step == cfg.steps):
            hist["eval"].append({"step": step, **eval_fn(model)})
    seconds = time.time() - t0
    if checkpoint_path:
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}   # loads anywhere
        torch.save({"state_dict": state, "history": hist, "device": device}, checkpoint_path)
    return TrainResult(checkpoint_path, hist, seconds)
