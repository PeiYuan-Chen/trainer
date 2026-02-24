from typing import Any

from hydra.utils import instantiate
from omegaconf import OmegaConf

from .trainer import Trainer


def train_func(config_dict: dict[str, Any]):
    cfg = OmegaConf.create(config_dict)

    trainer: Trainer = instantiate(cfg.trainer)

    trainer.train()
    trainer.close()
