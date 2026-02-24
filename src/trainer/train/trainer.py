import logging
import random
import tempfile
import posixpath
from abc import abstractmethod
from typing import Any, Callable, Iterable, TypeAlias

import numpy as np
import ray.train
import ray.train.torch
import torch
from torch._utils import _get_device_module
from torch.utils.data import Dataset, DataLoader
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from trainer.components import Checkpointer, Logger
from trainer.distributed import ParallelDims, dist_mean, dist_max
from trainer.utils import set_determinism, maybe_enable_amp, clip_grad_norm_
from .config import TrainingConfig, ParallelConfig

logger = logging.getLogger(__name__)

ParamsT: TypeAlias = (
    Iterable[torch.Tensor]
    | Iterable[dict[str, Any]]
    | Iterable[tuple[str, torch.Tensor]]
)


class Trainer(Stateful):
    def __init__(
        self,
        *,
        model_factory: Callable[[], torch.nn.Module],
        optimizer_factory: Callable[[ParamsT], torch.optim.Optimizer],
        lr_scheduler_factory: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler
        ],
        dataset: Dataset,
        metric_logger: Logger,
        checkpointer: Checkpointer,
        training_config: TrainingConfig,
        parallel_config: ParallelConfig | None = None,
        lora_config: LoraConfig | None = None,
    ):
        self.training_config = training_config
        self.lora_config = lora_config

        if self.training_config.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        # device
        self.device = ray.train.torch.get_device()
        device_module = _get_device_module(self.device.type)
        device_module.set_device(self.device)

        # device mesh
        parallel_config = parallel_config or ParallelConfig()
        self.parallel_dims = ParallelDims(
            dp_replicate=parallel_config.dp_replicate,
            dp_shard=parallel_config.dp_shard,
            cp=parallel_config.cp,
            world_size=ray.train.get_context().get_world_size(),
        )
        self.parallel_dims.build_mesh()
        if self.parallel_dims.dp_enabled:
            batch_mesh = self.parallel_dims.get_mesh("batch")
            batch_degree, batch_rank = batch_mesh.size(), batch_mesh.get_local_rank()
        else:
            batch_degree, batch_rank = 1, 0

        # determinism
        seed = set_determinism(
            device=self.device,
            world_size=ray.train.get_context().get_world_size(),
            distinct_seed_meshes=list(
                filter(
                    lambda mesh: mesh is not None,
                    [
                        self.parallel_dims.get_optional_mesh(dim)
                        for dim in self.training_config.distinct_seed_mesh_dims
                    ],
                )
            ),
            seed=self.training_config.seed,
            deterministic=self.training_config.deterministic,
            deterministic_warn_only=self.training_config.deterministic_warn_only,
        )

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(seed)

        self.dataloader = DataLoader(
            dataset,
            batch_size=self.training_config.batch_size,
            num_workers=self.training_config.num_workers,
            pin_memory=self.training_config.pin_memory,
            persistent_workers=self.training_config.persistent_workers,
            prefetch_factor=self.training_config.prefetch_factor,
            drop_last=self.training_config.drop_last,
            generator=g,
            worker_init_fn=seed_worker,
        )

        model = model_factory()
        if lora_config is not None:
            model.requires_grad_(False)
            model = get_peft_model(model, lora_config)
        model = self.parallelize_model(model, self.parallel_dims)
        model.train()
        self.model = model
        self.optimizer = optimizer_factory(
            [p for p in model.parameters() if p.requires_grad]
        )
        self.lr_scheduler = lr_scheduler_factory(self.optimizer)

        self.checkpointer = checkpointer
        self.logger = metric_logger

        # train context
        self.maybe_enable_amp = maybe_enable_amp(
            self.parallel_dims.fsdp_enabled,
            mixed_precision_param=self.training_config.mixed_precision_param,
        )

        # trainer state
        self.step = 0

        logger.info(
            f"trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

    def train(self) -> None:
        self.load_checkpoint()

        data_iterator = self.batch_generator(self.dataloader)
        while self.step < self.training_config.steps:
            self.step += 1
            logs = self.train_step(data_iterator)
            self.log(logs)
            self.save_checkpoint()

    def close(self) -> None:
        if hasattr(self, "logger") and self.logger:
            self.logger.close()

    def train_step(
        self, data_iterator: Iterable[dict[str, torch.Tensor]]
    ) -> dict[str, Any]:
        self.optimizer.zero_grad(set_to_none=True)
        # Save the current step learning rate for logging
        lr = self.lr_scheduler.get_last_lr()[0]

        accumulated_losses = []
        for _microbatch in range(self.training_config.gradient_accumulation_steps):
            batch = next(data_iterator)
            with self.maybe_enable_amp:
                loss = self.forward_step(batch)
                loss = loss / self.training_config.gradient_accumulation_steps
            loss.backward()
            accumulated_losses.append(loss.detach())

        grad_norm = clip_grad_norm_(
            [
                p
                for p in self.model.parameters()
                if p.requires_grad and p.grad is not None
            ],
            self.training_config.max_norm,
            foreach=True,
        )
        self.optimizer.step()
        self.lr_scheduler.step()

        loss = torch.sum(torch.stack(accumulated_losses))

        return {
            "loss": loss,
            "grad_norm": grad_norm.item(),
            "lr": lr,
        }

    def state_dict(self) -> dict[str, Any]:
        model_state_dict = get_model_state_dict(self.model)
        if self.lora_config is not None:
            model_state_dict = get_peft_model_state_dict(self.model, model_state_dict)
        if self.training_config.save_model_only:
            return {
                "model": model_state_dict,
            }
        optimizer_state_dict = get_optimizer_state_dict(self.optimizer)
        return {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
            "scheduler": self.lr_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        if self.lora_config is not None:
            set_peft_model_state_dict(self.model, state_dict["model"])
        else:
            set_model_state_dict(self.model, state_dict["model"])
        if self.training_config.save_model_only:
            return
        set_optimizer_state_dict(
            self.model,
            self.optimizer,
            optim_state_dict=state_dict["optimizer"],
        )
        self.lr_scheduler.load_state_dict(state_dict["scheduler"])

    def batch_generator(
        self, data_iterable: Iterable[dict[str, torch.Tensor]]
    ) -> Iterable[dict[str, torch.Tensor]]:
        while True:  # infinite loop
            data_iterator = iter(data_iterable)
            for batch in data_iterator:
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.cuda(non_blocking=True)
                yield batch

    def log(self, raw_logs: dict[str, Any]) -> None:
        if not self._should_log():
            return

        logs = {}
        for k, v in raw_logs.items():
            if torch.is_tensor(v):
                if self.parallel_dims.dp_cp_enabled:
                    v = v.detach()
                    loss_mesh = self.parallel_dims.get_optional_mesh("loss")
                    logs[f"global_avg_{k}"] = dist_mean(v, loss_mesh)
                    logs[f"global_max_{k}"] = dist_max(v, loss_mesh)
                else:
                    v = v.detach().item()
                    logs[k] = v
            else:
                logs[k] = v

        logger.info({"step": self.step, **logs})
        self.logger.log(self.step, logs)

    def save_checkpoint(self) -> None:
        if not self._should_save_checkpoint():
            return

        temp_dir = tempfile.mkdtemp()
        self.checkpointer.save(self, temp_dir)
        checkpoint = ray.train.Checkpoint.from_directory(temp_dir)
        checkpoint.set_metadata({"step": self.step})
        ray.train.report(
            metrics={"step": self.step},
            checkpoint=checkpoint,
            checkpoint_dir_name=posixpath.join("checkpoints", f"step_{self.step:08d}"),
            checkpoint_upload_mode=ray.train.CheckpointUploadMode.ASYNC,
            delete_local_checkpoint_after_upload=True,
        )

    def load_checkpoint(self, checkpoint_dir: str | None = None) -> None:
        checkpoint = (
            ray.train.get_checkpoint()
            if checkpoint_dir is None
            else ray.train.Checkpoint.from_directory(checkpoint_dir)
        )
        if checkpoint:
            self.step = checkpoint.get_metadata()["step"]
            with checkpoint.as_directory() as checkpoint_dir:
                self.checkpointer.load(self, checkpoint_dir)
                logger.info(
                    f"Successfully loaded checkpoint from from {checkpoint_dir} for resuming training at step {self.step}"
                )

    def _should_log(self) -> bool:
        return self.step == 1 or self.step % self.training_config.log_freq == 0

    def _should_save_checkpoint(self) -> bool:
        if self.step == 1 and self.training_config.enable_first_step_checkpoint:
            return True

        if self.step == self.training_config.steps:
            return True

        if self.step % self.training_config.checkpoint_freq == 0:
            return True

        return False

    @abstractmethod
    def forward_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor: ...

    @staticmethod
    def parallelize_model(
        model: torch.nn.Module,
        parallel_dims: ParallelDims,
    ) -> torch.nn.Module:
        return model.cuda()
