"""pa — Self-evolving Pydantic-AI agent harness."""

from pa.manifest import Manifest, Registration, ManifestError, CardinalityError
from pa.runtime import build_agent
from pa.capability import PaRegistrations

__version__ = "0.1.0"
__all__ = [
    "Manifest",
    "Registration",
    "ManifestError",
    "CardinalityError",
    "PaRegistrations",
    "build_agent",
    "__version__",
]
