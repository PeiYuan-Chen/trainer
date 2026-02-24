import torch
from torch.distributed.tensor import DTensor
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch.distributed.device_mesh import DeviceMesh
from torch import distributed as dist


def _dist_reduce(
    x: torch.Tensor,
    reduceOp: str,
    mesh: DeviceMesh | None,
    extra_pg: dist.ProcessGroup | None,
) -> float:
    """Perform distributed reduction on a tensor.

    Args:
        x (torch.Tensor): Input tensor.
        reduceOp (str): Reduce operation to perform.
        mesh (DeviceMesh | None): Device mesh to use for reduction.
            If None, no reduction is performed but simply convert the tensor to a float.
        extra_pg (dist.ProcessGroup, optional): Extra process group to use for reduction.
            Defaults to None. If provided, this all_reduce will be called for the extra
            process group, and then the result will be all_reduced for the mesh.
    """
    if isinstance(x, DTensor):
        # functional collectives do not support DTensor inputs
        x = x.full_tensor()

    if extra_pg is not None:
        x = funcol.all_reduce(x, reduceOp=reduceOp, group=extra_pg)

    if mesh is None:
        return x.item()

    assert x.numel() == 1  # required by `.item()`
    return funcol.all_reduce(x, reduceOp=reduceOp, group=mesh).item()


# TODO: rename this to maybe_dist_max
def dist_max(
    x: torch.Tensor,
    mesh: DeviceMesh | None = None,
    extra_pg: dist.ProcessGroup | None = None,
) -> float:
    return _dist_reduce(
        x, reduceOp=c10d.ReduceOp.MAX.name, mesh=mesh, extra_pg=extra_pg
    )


def dist_sum(
    x: torch.Tensor,
    mesh: DeviceMesh | None = None,
    extra_pg: dist.ProcessGroup | None = None,
) -> float:
    return _dist_reduce(
        x, reduceOp=c10d.ReduceOp.SUM.name, mesh=mesh, extra_pg=extra_pg
    )


def dist_mean(
    x: torch.Tensor,
    mesh: DeviceMesh | None = None,
    extra_pg: dist.ProcessGroup | None = None,
) -> float:
    return _dist_reduce(
        x, reduceOp=c10d.ReduceOp.AVG.name, mesh=mesh, extra_pg=extra_pg
    )
