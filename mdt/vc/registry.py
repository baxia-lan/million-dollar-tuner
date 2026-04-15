"""Backend discovery and selection for voice conversion."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import VCBackend

logger = logging.getLogger(__name__)

_BACKENDS: dict[str, type] = {}


def _register_builtins() -> None:
    """Lazily register all built-in backends."""
    if _BACKENDS:
        return

    from .applio import ApplioBackend
    from .seedvc import SeedVCBackend
    from .sovits import SoVITSBackend

    _BACKENDS["applio"] = ApplioBackend
    _BACKENDS["sovits"] = SoVITSBackend
    _BACKENDS["seedvc"] = SeedVCBackend


def get_backend(name: str) -> "VCBackend":
    """Return an instance of the named backend."""
    _register_builtins()
    if name not in _BACKENDS:
        raise KeyError(f"Unknown backend '{name}'. Available: {list(_BACKENDS.keys())}")
    return _BACKENDS[name]()


def list_backends() -> list[str]:
    """Return names of all registered backends."""
    _register_builtins()
    return list(_BACKENDS.keys())


def get_available_backends() -> list[str]:
    """Return names of backends that are available on this system."""
    _register_builtins()
    available = []
    for name, cls in _BACKENDS.items():
        try:
            backend = cls()
            if backend.is_available():
                available.append(name)
        except Exception as e:
            logger.debug("Backend %s unavailable: %s", name, e)
    return available


def get_best_backend(profile_dir=None) -> "VCBackend":
    """Return the best available backend, preferring Applio > SoVITS > Seed-VC."""
    _register_builtins()
    preference = ["applio", "sovits", "seedvc"]
    for name in preference:
        try:
            backend = _BACKENDS[name]()
            if backend.is_available():
                return backend
        except Exception:
            continue
    raise RuntimeError("No voice conversion backend is available")
