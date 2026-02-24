import logging
from dataclasses import dataclass, field

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

logger = logging.getLogger(__name__)


@dataclass
class ParallelDims:
    dp_replicate: int = 1
    dp_shard: int = 1
    cp: int = 1
    world_size: int = 1

    _meshes: dict[str, DeviceMesh] = field(default_factory=dict)
    _world_mesh: DeviceMesh | None = None

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp_replicate, dp_shard, cp = (
            self.dp_replicate,
            self.dp_shard,
            self.cp,
        )
        for d in (dp_replicate, cp):
            assert d >= 1, "Parallelism degree should be >= 1, except for dp_shard"

        assert dp_shard == -1 or dp_shard >= 1, "dp_shard must -1 or >=1."
        if dp_shard < 0:
            self.dp_shard = dp_shard = self.world_size // (dp_replicate * cp)
        assert dp_shard >= 1

        assert dp_replicate * dp_shard * cp == self.world_size, (
            f"Invalid parallel dims: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * "
            f"cp({cp}) != WORLD_SIZE({self.world_size})"
        )

    def _mesh_exist(self, name: str, degree: int) -> bool:
        return degree > 1

    def build_mesh(self) -> DeviceMesh:
        def unflatten_mesh(
            world_mesh: DeviceMesh,
            dim_names: tuple[str, ...],
            dim_degrees: tuple[int, ...],
        ):
            """Unflatten the world mesh to create the required mesh dimensions.

            Uses fake backend for dimensions with degree 1 or for 'batch' dimension
            to avoid unnecessary process group creation.
            """
            backend_override = {}
            for name, degree in zip(dim_names, dim_degrees, strict=True):
                if (not self._mesh_exist(name, degree)) or name == "batch":
                    backend_override[name] = "fake"

            return world_mesh._unflatten(
                0, dim_degrees, dim_names, backend_override=backend_override
            )

        logger.info(
            f"Building device mesh with parallelism: "
            f"dp_replicate={self.dp_replicate}, dp_shard={self.dp_shard}, cp={self.cp}"
        )

        batch = self.dp_replicate * self.dp_shard
        fsdp = self.dp_shard * self.cp

        self._world_mesh = init_device_mesh(
            "cuda", (self.world_size,), mesh_dim_names=("world",)
        )
        dataloading_mesh = unflatten_mesh(
            self._world_mesh,
            ("batch", "cp"),
            (batch, self.cp),
        )
        loss_mesh = dataloading_mesh["batch", "cp"]._flatten("loss_mesh")
        dense_mesh = unflatten_mesh(
            self._world_mesh, ("dp_replicate", "fsdp"), (self.dp_replicate, fsdp)
        )

        self._global_meshes = {
            "dataloading": dataloading_mesh,
            "loss": loss_mesh,
            "dense": dense_mesh,
        }

        self._meshes = {
            "batch": dataloading_mesh["batch"],
            "loss": loss_mesh["loss"],
            "dp_replicate": dense_mesh["dp_replicate"],
            "fsdp": dense_mesh["fsdp"],
            "cp": dataloading_mesh["cp"],
        }

        # Validate mesh sizes
        self._validate_meshes()

        logger.info(
            f"Successfully created meshes with active dimensions: "
            f"{list(self.get_all_one_dimensional_meshes().keys())}"
        )

        return self._world_mesh

    def _validate_meshes(self):
        """Validate that created meshes have the expected sizes."""
        expected_sizes = {
            "batch": self.dp_replicate * self.dp_shard,
            "loss": self.dp_replicate * self.dp_shard * self.cp,
            "dp_replicate": self.dp_replicate,
            "fsdp": self.dp_shard * self.cp,
            "cp": self.cp,
        }

        for mesh_name, expected_size in expected_sizes.items():
            actual_size = self._meshes[mesh_name].size()
            assert actual_size == expected_size, (
                f"Mesh '{mesh_name}' has unexpected size: "
                f"expected {expected_size}, got {actual_size}"
            )

    def get_optional_mesh(self, dims: str | list[str]) -> DeviceMesh | None:
        """Get a device mesh by dimension name(s), returning None if not enabled.

        Args:
            dims: Names of the mesh dimension. Valid options include:
                 'pp', 'batch', 'loss', 'dp_replicate', 'fsdp',
                 'cp', 'tp', 'ep', 'etp', 'efsdp'.

        Returns:
            DeviceMesh for the requested dimension(s), or None if:
            - The dimension size is 1 (parallelism not enabled)
            - The dimension doesn't exist (except efsdp which can exist even if size is 1 when ep > 1)

        Raises:
            ValueError: If the requested dimension name(s) is not valid.
        """
        if not self._meshes:
            self.build_mesh()

        if isinstance(dims, str):
            dims = [dims]

        for mesh_name in dims:
            if mesh_name not in self._meshes:
                raise ValueError(
                    f"Invalid mesh dim: '{mesh_name}'. "
                    f"Valid dimensions are: {list(self._meshes.keys())}"
                )

        if any(not self._mesh_exist(dim, self._meshes[dim].size()) for dim in dims):
            return None

        if len(dims) == 1:
            return self._meshes[dims[0]]
        else:
            for global_mesh in self._global_meshes.values():
                assert global_mesh.mesh_dim_names is not None
                if not set(dims).issubset(set(global_mesh.mesh_dim_names)):
                    continue
                return global_mesh[tuple(dims)]
            raise ValueError(f"Invalid mesh name combinations {dims}.")

    def get_mesh(self, dims: str | list[str]) -> DeviceMesh:
        """Get a device mesh by dimension name(s), raising if not available.

        Args:
            dims: Names of the mesh dimension. Valid options include:
                 'pp', 'batch', 'loss', 'dp_replicate', 'fsdp',
                 'cp', 'tp', 'ep', 'etp', 'efsdp'.

        Returns:
            DeviceMesh for the requested dimension(s).

        Raises:
            ValueError: If the mesh is not available (dimension size = 1 or not enabled),
                or if the requested dimension name(s) is not valid.
        """
        mesh = self.get_optional_mesh(dims)
        if mesh is None:
            enabled_str = (
                "enabled (size > 1)" if isinstance(dims, str) else "all enabled"
            )
            raise ValueError(
                f"Mesh '{dims}' is not available. "
                f"Ensure the corresponding parallelism dimension is {enabled_str}."
            )
        return mesh

    def get_all_one_dimensional_meshes(self) -> dict[str, DeviceMesh]:
        """Get all enabled one-dimensional device meshes.

        Returns a dictionary of enabled one-dimensional device meshes, allowing you to
        access their process groups.

        Note:
            Device meshes created with the Fake backend are still included in the results.

        Returns:
            dict[str, DeviceMesh]: A dictionary mapping mesh dimension names to their
                corresponding DeviceMesh objects. Only includes meshes where:
                - ndim == 1 (one-dimensional)
                - parallelism is enabled (size > 1)

        Example:
            >>> parallel_dims = ParallelDims(
            ...     dp_replicate=2, dp_shard=2, cp=1, tp=2, pp=1, ep=1, etp=1, world_size=8
            ... )
            >>> meshes = parallel_dims.get_all_one_dimensional_meshes()
            >>> print(meshes.keys())
            dict_keys(['dp_replicate', 'fsdp', 'tp', 'batch', 'loss', 'efsdp'])
        """
        if not self._meshes:
            self.build_mesh()
        return {k: v for k, v in self._meshes.items() if v.ndim == 1 and v.size() > 1}

    @property
    def world_mesh(self) -> DeviceMesh:
        if self._world_mesh is None:
            self._world_mesh = self.build_mesh()
        return self._world_mesh

    @property
    def dp_enabled(self):
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self):
        return self.dp_shard > 1

    @property
    def cp_enabled(self):
        return self.cp > 1

    @property
    def dp_cp_enabled(self):
        return self.dp_enabled or self.cp_enabled

    @property
    def fsdp_enabled(self):
        return self.dp_shard_enabled or self.cp_enabled
