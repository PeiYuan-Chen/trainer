from dataclasses import dataclass


@dataclass
class RayTrainConfig:
    name: str | None = None
    num_workers: int = 1
    use_gpu: bool = False
    resources_per_worker: dict[str, int] | None = None
    storage_path: str | None = None
    num_to_keep: int | None = None
