import posixpath
import torch
from torch.distributed.checkpoint.stateful import Stateful

from .base import Checkpointer


class TorchSaveCheckpointer(Checkpointer):
    @torch.no_grad()
    def save(self, state_obj: Stateful, checkpoint_dir: str) -> None:
        torch.save(
            state_obj.state_dict(), posixpath.join(checkpoint_dir, "state_dict.pt")
        )

    @torch.no_grad()
    def load(self, state_obj: Stateful, checkpoint_dir: str) -> None:
        state_obj.load_state_dict(
            torch.load(posixpath.join(checkpoint_dir, "state_dict.pt"))
        )
