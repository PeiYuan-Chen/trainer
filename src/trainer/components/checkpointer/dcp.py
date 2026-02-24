import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful

from .base import Checkpointer


class DistributedCheckpointer(Checkpointer):
    @torch.no_grad()
    def save(self, state_obj: Stateful, checkpoint_dir: str) -> None:
        state_dict = {"app": state_obj}
        dcp.save(state_dict=state_dict, checkpoint_id=checkpoint_dir)

    @torch.no_grad()
    def load(self, state_obj: Stateful, checkpoint_dir: str) -> None:
        state_dict = {"app": state_obj}
        dcp.load(state_dict=state_dict, checkpoint_id=checkpoint_dir)
