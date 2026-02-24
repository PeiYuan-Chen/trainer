from typing import Protocol

import torch
from torch.distributed.checkpoint.stateful import Stateful


class Checkpointer(Protocol):
    @torch.no_grad()
    def save(self, state_obj: Stateful, checkpoint_dir: str) -> None: ...

    @torch.no_grad()
    def load(self, state_obj: Stateful, checkpoint_dir: str) -> None: ...
