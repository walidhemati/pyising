"""pyising: roofline-optimised Ising Monte Carlo on the GPU from Python."""

from .backends import BACKENDS, REGISTRY, available, create, load
from .exact import BETA_C_2D, BETA_C_3D, TC_2D, TC_3D

__version__ = "0.1.0"

__all__ = ["BACKENDS", "REGISTRY", "available", "create", "load", "BETA_C_2D", "BETA_C_3D",
           "TC_2D", "TC_3D", "__version__"]
