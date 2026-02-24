from dataclasses import dataclass


@dataclass
class ParallelConfig:
    dp_replicate: int = 1
    dp_shard: int = -1
    cp: int = 1
