from pathlib import Path
from typing import Sequence

import torch
import numpy as np
from streaming import Stream


def make_streams(
    remote: str | list[str] | None = None,
    local: str | list[str] | None = None,
    proportion: list | None = None,
    repeat: list | None = None,
    choose: list | None = None,
) -> list[Stream]:
    """Helper function to create a list of Stream objects from a set of remotes and stream weights.

    Args:
        remote (Union[str, Sequence[str]]): The remote path or paths to stream from.
        local (Union[str, Sequence[str]], optional): The local path or paths to cache the data. If not provided, the
            default local path is used. Default: ``None``.
        proportion (list, optional): Specifies how to sample this Stream relative to other Streams. Default: ``None``.
        repeat (list, optional): Specifies the degree to which a Stream is upsampled or downsampled. Default: ``None``.
        choose (list, optional): Specifies the number of samples to choose from a Stream. Default: ``None``.

    Returns:
        List[Stream]: A list of Stream objects.
    """
    remote, local = _make_remote_and_local_sequences(remote, local)
    proportion, repeat, choose = _make_weighting_sequences(
        len(remote) if remote else len(local), proportion, repeat, choose
    )

    return [
        Stream(remote=r, local=l, proportion=p, repeat=rt, choose=c)
        for r, l, p, rt, c in zip(remote, local, proportion, repeat, choose)
    ]


def _make_remote_and_local_sequences(
    remote: str | list[str] | None = None,
    local: str | list[str] | None = None,
) -> tuple[list[str], list[str]]:
    if remote is None and local is None:
        raise ValueError("remote and local must be provided")
    if isinstance(remote, str):
        remote = [remote]
    if isinstance(local, str):
        local = [local]
    if local and not remote:
        remote = [None] * len(local)
    if remote and not local:
        local = [_make_default_local_path(r) for r in remote]

    if isinstance(remote, Sequence) and isinstance(local, Sequence):
        if len(remote) != len(local):
            raise ValueError(
                f"remote and local Sequences must be the same length, got lengths {len(remote)} and {len(local)}"
            )
    return remote, local
    # else:
    #     raise ValueError(
    #         f"remote and local must be both Strings or Sequences, got types {type(remote)} and {type(local)}."
    #     )


def _make_default_local_path(remote_path: str, base: str = "/tmp") -> str:
    p = Path(remote_path)
    if p.is_absolute():
        # remove anchor(e.g. '/' or 'C:\\')
        p = p.relative_to(p.anchor)
    return str(Path(base) / p)


def _make_weighting_sequences(
    n: int,
    proportion: list | None = None,
    repeat: list | None = None,
    choose: list | None = None,
) -> tuple[list[float | None], list[int | None], list[int | None]]:
    def _normalize(name, seq):
        if seq is None:
            return [None] * n
        if len(seq) != n:
            raise ValueError(
                f"{name} sequence must be the same length as remote, got length {len(seq)}"
            )
        return seq

    proportion = _normalize("proportion", proportion)
    repeat = _normalize("repeat", repeat)
    choose = _normalize("choose", choose)

    return proportion, repeat, choose


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().contiguous()

    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.uint16)

    return tensor.numpy()


def numpy_to_tensor(arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    t = torch.from_numpy(arr.copy())  # copy to avoid non-writeable error
    if dtype == torch.bfloat16:
        if t.dtype != torch.uint16:
            raise ValueError(f"Expected uint16 dtype, got {t.dtype}")
        t = t.view(torch.bfloat16)
    return t


# def storage_numpy_to_bytes(arr: np.ndarray) -> bytes:
#     if not arr.flags.c_contiguous:
#         arr = np.ascontiguousarray(arr)
#     return arr.tobytes(order="C")


# def decode_storage_bytes_to_tensor(
#     data: bytes,
#     shape: tuple[int, ...],
#     dtype: torch.dtype = torch.bfloat16,
# ) -> torch.Tensor:
#     assert dtype in _TORCH_TO_STORAGE_NUMPY, f"Unsupported dtype: {dtype}"

#     arr = np.frombuffer(data, dtype=_TORCH_TO_STORAGE_NUMPY[dtype]).reshape(shape)
#     return torch.from_numpy(arr).view(dtype)
