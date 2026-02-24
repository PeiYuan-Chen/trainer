from typing import Any

from .base import Logger


class TensorBoardLogger(Logger):
    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=log_dir)

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def close(self) -> None:
        self.writer.close()
