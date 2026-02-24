import os
import random
import logging

import numpy as np
import torch
import torch.distributed.distributed_c10d as c10d
from torch.distributed.device_mesh import DeviceMesh

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # The native RNGs and python RNG may not be important, except for the 1-D PP case, but we seed them for consistency.
    torch.manual_seed(seed)
    # PYTHONHASHSEED can be a decimal number in the range [0, 2**32 - 1]
    os.environ["PYTHONHASHSEED"] = str(seed % 2**32)


def set_determinism(
    device: torch.device,
    world_size: int,
    *,
    distinct_seed_meshes: list[DeviceMesh] | None = None,
    seed: int | None = None,
    deterministic: bool = False,
    deterministic_warn_only: bool = False,
) -> int:
    """
    Set the same DTensor manual seed for all dimensions in world mesh, but only different seeds
    across dimensions denoted by `distinct_seed_mesh_dims`. An example use case is pipeline parallelism,
    where we want to have the same seed across SPMD groups, but different seeds across PP groups.

    Currently, does not set seeds for the CUDA RNG since TorchTitan always uses DTensor for SPMD parallelisms,
    and DTensor manages its own RNG tracker, but we could extend to support both if needed.

    Set Determinism flags for increased reproducibility with loss of performance.

    Args:
        world_mesh: Device mesh for distributed training
        device: Device to use
        debug_config: Debug config to use
        distinct_seed_mesh_dims: List of mesh dimension names to have distinct seeds across.
    """
    if deterministic:
        logger.info("Deterministic algorithm enabled (expect perf degradation).")
        torch.use_deterministic_algorithms(True, warn_only=deterministic_warn_only)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # env var for deterministic CuBLAS
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if seed is None:
        seed_tensor = torch.get_rng_state()[:8].to(device)
        if world_size > 1:
            torch.distributed.broadcast(seed_tensor, src=0)
        seed = seed_tensor.to("cpu").view(torch.uint64).item()

    # Set distinct seed for each rank in mesh dimensions, with dimension names provided by `distinct_seed_mesh_dims`
    # For PP + SPMD cases, we want to separate the world into the SPMD mesh and the PP mesh,
    # and choose a unique seed for each rank on the PP mesh.
    # We support multiple distinct dimensions by adding each distinct dimension's local rank to the seed.
    # distinct_seed_meshes = [
    #     parallel_dims.get_optional_mesh(dim) for dim in distinct_seed_mesh_dims
    # ]
    # distinct_seed_meshes = [mesh for mesh in distinct_seed_meshes if mesh is not None]
    # assert all(mesh is not None for mesh in distinct_seed_meshes)
    distinct_seed_meshes = list(
        filter(lambda mesh: mesh is not None, distinct_seed_meshes)
    )
    if distinct_seed_meshes:
        # Each dimension contributes: local_rank * (product of all previous dimension sizes)
        # This guarantees uniqueness like multi-dimensional array indexing
        seed_offset = 0
        cumulative_size = 1

        for distinct_mesh in distinct_seed_meshes:
            local_rank = distinct_mesh.get_local_rank()
            # Add contribution from this dimension
            seed_offset += local_rank * cumulative_size
            # Update cumulative size for next dimension
            cumulative_size *= distinct_mesh.size()

        seed += seed_offset
        seed %= 2**64

        logger.debug(
            f"Distinct dims {', '.join([mesh.mesh_dim_names[0] for mesh in distinct_seed_meshes])}, Global rank {c10d.get_rank()} using seed: {seed}"
        )
    else:
        logger.debug(f"Global Rank {c10d.get_rank()} using seed: {seed}")

    set_seed(seed)
    return seed
