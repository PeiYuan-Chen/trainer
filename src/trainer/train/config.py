from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TrainingConfig:
    steps: int
    allow_tf32: bool = True
    seed: int | None = None
    deterministic: bool = False
    deterministic_warn_only: bool = False
    distinct_seed_mesh_dims: list[str] = field(default_factory=list)
    mixed_precision_param: Literal["bfloat16", "float32"] = "bfloat16"
    gradient_accumulation_steps: int = 1
    max_norm: float = 1.0

    # dataloader
    batch_size: int = 1
    num_workers: int = 16
    drop_last: bool = True
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    enable_first_step_checkpoint: bool = True
    checkpoint_freq: int = 500
    log_freq: int = 1
    save_model_only: bool = False
