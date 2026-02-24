import contextlib
import logging
from typing import Literal

import torch

logger = logging.getLogger(__name__)


TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def maybe_enable_amp(
    fsdp_enabled: bool,
    mixed_precision_param: Literal["bfloat16", "float32"] = "bfloat16",
    device_type: str = "cuda",
) -> contextlib.AbstractContextManager[None]:
    if fsdp_enabled:
        # FSDP handles mixed precision internally
        logger.info("Mixed precision training is handled by fully_shard")
        return contextlib.nullcontext()
    else:
        # the following code will only be executed for DDP or single-device training
        logger.info("Mixed precision training is handled by AMP")
        # pyrefly: ignore [bad-return]
        return torch.autocast(
            device_type,
            dtype=TORCH_DTYPE_MAP[mixed_precision_param],
        )
