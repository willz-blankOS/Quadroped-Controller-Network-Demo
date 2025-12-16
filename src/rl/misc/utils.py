from dataclasses import dataclass
from typing import Callable

from rl.policy.policy import Curriculum


@dataclass
class TrainingPreset:
    curriculum: Curriculum
    target_generating_fn: Callable