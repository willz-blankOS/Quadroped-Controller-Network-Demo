import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp
import flax.nnx as nnx

import orbax.checkpoint as ocp
from orbax.checkpoint import CheckpointManager

from rl.nn.model import ControllerNet

@jax.tree_util.register_dataclass
@dataclass
class TrainState:
    optimizer_state: nnx.State = None
    model_state: nnx.State = None
    steps: int = 0


@dataclass
class PolicyCheckpointConfig:
    save_every_steps: int = 10_000
    save_best_metric: bool = True
    metric_name: str = "avg_return"


class PolicyCheckpointManager:
    def __init__(
        self,
        directory,
        cfg: PolicyCheckpointConfig = None,
        checkpointers = None,
        options = None,
    ):
        self.cfg = cfg or PolicyCheckpointConfig()

        self.options = ocp.CheckpointManagerOptions(
            max_to_keep=5, create=True,
            enable_async_checkpointing=True
        )

        self.manager = CheckpointManager(
            directory, checkpointers, options, 
            item_names=("train_state", "meta")
        )

        self.best_metric: Optional[float] = None
        self.best_step: Optional[int] = None

    def maybe_restore(self) -> tuple[Optional[TrainState], dict]:
        latest_step = self.manager.latest_step()
        if latest_step is None:
            print("[CKPT] No checkpoint found, starting fresh")
            return None, {}
        
        restored = self.manager.restore(latest_step)
        meta = restored.get(meta, {})
        self.best_metric = meta.get("best_metric", None)
        self.best_step = meta.get("best_step", None)
        print(f"[CKPT] Restored from step {latest_step}, best_metric={self.best_metric}")
        return restored["train_state"], meta

    def _save(self, step: int, train_state: TrainState, meta: dict):
        to_save = {
            "train_state": train_state,
            "meta": meta
        }
        print(f"[CKPT] Saving at step {step}")
        self.manager.save(step, to_save)

    def maybe_save_periodic(self, train_state: TrainState, meta: dict):
        step = train_state.global_step
        if step % self.cfg.save_every_steps == 0:
            self._save(step, train_state, meta)

    def maybe_save_best(self, train_state: TrainState, meta: dict, metric_value: float):
        if not self.cfg.save_best_metric:
            return
        
        step = train_state.global_step
        if (self.best_metric is None) or (metric_value > self.best_metric):
            print(f"[CKPT] New best {self.cfg.metric_name}={metric_value} at step {step}")
            self.best_metric = metric_value
            self.best_step = step
            meta = dict(meta)
            meta["best_metric"] = float(metric_value)
            meta["best_step"] = int(step)
            self._save(step, train_state, meta)
