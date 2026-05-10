"""Hot-pluggable backend interfaces."""

from testbed.backends.base import Backend
from testbed.backends.real import RealBackend, RealExcavatorBackend

__all__ = ["Backend", "RealBackend", "RealExcavatorBackend"]
