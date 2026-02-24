import os

os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

from typing import Any
import logging
from dataclasses import dataclass, field

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING
import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig

from .config import RayTrainConfig
from .train import train_func, TrainingConfig

logger = logging.getLogger(__name__)
cs = ConfigStore.instance()


@dataclass
class Config:
    ray_train: RayTrainConfig = field(default_factory=RayTrainConfig)
    trainer: Any = MISSING


cs.store(name="base_config", node=Config)
cs.store(group="trainer/training_config", name="base", node=TrainingConfig)


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: Config):
    ray.init()
    scaling_config = ScalingConfig(
        num_workers=cfg.ray_train.num_workers,
        use_gpu=cfg.ray_train.use_gpu,
        resources_per_worker=cfg.ray_train.resources_per_worker,
    )
    checkpoint_config = CheckpointConfig(
        num_to_keep=cfg.ray_train.num_to_keep,
    )
    run_config = RunConfig(
        name=cfg.ray_train.name,
        storage_path=cfg.ray_train.storage_path,
        checkpoint_config=checkpoint_config,
    )  # stoage_path/name/

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=OmegaConf.to_container(cfg.trainer, resolve=True),
        scaling_config=scaling_config,
        run_config=run_config,
    )
    result = trainer.fit()
