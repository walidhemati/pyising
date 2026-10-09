"""Common interface for every backend/variant."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from ..lattice import (
    check_shape,
    energy_per_spin,
    magnetization,
    random_lattice,
    threshold_table,
)


class IsingSimulation(ABC):
    """A checkerboard Ising Monte Carlo simulation living on one backend.

    Parameters
    ----------
    shape: ``(L, L)`` or ``(L, L, L)``; extents even, last extent a multiple of
        :attr:`layout_multiple`.
    beta: inverse temperature ``1 / T`` (``J = k_B = 1``).
    seed: 64-bit Philox key.
    algorithm: ``"metropolis"`` or ``"heatbath"``.
    init: ``"hot"`` (i.i.d. random), ``"cold"`` (all +1) or an explicit int8 lattice.
    """

    backend: ClassVar[str] = "base"
    variant: ClassVar[str] = "base"
    dims: ClassVar[tuple[int, ...]] = (2, 3)
    algorithms: ClassVar[tuple[str, ...]] = ("metropolis", "heatbath")
    layout_multiple: ClassVar[int] = 8
    #: Random-stream family: simulations with the same stream, seed and start are bit-identical.
    stream: ClassVar[str] = "int8"
    #: Minimal DRAM traffic per attempted spin update (read own + read neighbours + write own).
    bytes_per_update: ClassVar[float] = 3.0
    #: Whether the backend runs on the GPU.
    gpu: ClassVar[bool] = True

    def __init__(
        self,
        shape: tuple[int, ...],
        beta: float,
        seed: int = 0,
        algorithm: str = "metropolis",
        init: str | np.ndarray = "hot",
    ) -> None:
        self.shape = check_shape(shape, self.layout_multiple)
        self.ndim = len(self.shape)
        if self.ndim not in self.dims:
            raise ValueError(f"{self.backend}/{self.variant} supports dims {self.dims}")
        if algorithm not in self.algorithms:
            raise ValueError(f"{self.backend}/{self.variant} supports {self.algorithms}")
        self.algorithm = algorithm
        self.seed = int(seed) & 0xFFFFFFFFFFFFFFFF
        self.beta = float(beta)
        self.t = 0
        self.rows = int(np.prod(self.shape[:-1]))
        self.lh = self.shape[-1] // 2
        self.table = threshold_table(self.beta, self.ndim, self.algorithm)
        self._setup()
        if isinstance(init, np.ndarray):
            self.set_lattice(init)
        elif init == "cold":
            self._fill_cold()
        elif init == "hot":
            self._fill_hot()
        else:
            raise ValueError(f"unknown init {init!r}")

    @property
    def n_spins(self) -> int:
        return int(np.prod(self.shape))

    def set_beta(self, beta: float) -> None:
        """Change the temperature (keeps the configuration and the sweep counter)."""
        self.beta = float(beta)
        self.table = threshold_table(self.beta, self.ndim, self.algorithm)
        self._upload_table()

    def sweep(self, n: int = 1) -> None:
        """Run ``n`` full checkerboard sweeps (colour 0 then colour 1)."""
        for _ in range(int(n)):
            self._half_sweep(0)
            self._half_sweep(1)
            self.t += 1

    # ---- hooks -------------------------------------------------------------------------------
    @abstractmethod
    def _setup(self) -> None:
        """Allocate device state and upload the acceptance table."""

    @abstractmethod
    def _half_sweep(self, color: int) -> None:
        """Update every site of one colour using sweep counter ``self.t``."""

    @abstractmethod
    def get_lattice(self) -> np.ndarray:
        """Return the configuration as a full int8 host array."""

    @abstractmethod
    def set_lattice(self, lat: np.ndarray) -> None:
        """Load a full int8 host configuration."""

    def _upload_table(self) -> None:  # noqa: B027 - optional hook
        """Push ``self.table`` to the device (backends that keep a device copy override)."""

    def _fill_hot(self) -> None:
        self.set_lattice(random_lattice(self.shape, self.seed))

    def _fill_cold(self) -> None:
        self.set_lattice(np.ones(self.shape, dtype=np.int8))

    def synchronize(self) -> None:  # noqa: B027 - optional hook
        """Block until all queued device work is done (no-op on the CPU)."""

    def magnetization(self) -> float:
        """Signed magnetisation per spin."""
        return magnetization(self.get_lattice())

    def energy(self) -> float:
        """Energy per spin."""
        return energy_per_spin(self.get_lattice())

    def observables(self) -> tuple[float, float]:
        """``(m, e)`` per spin; GPU backends override with an on-device reduction."""
        return self.magnetization(), self.energy()

    def device_bytes(self) -> int:
        """Bytes of lattice state held on the device."""
        return int(self.n_spins * self.bytes_per_update / 3.0)
