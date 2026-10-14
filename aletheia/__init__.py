"""Aletheia: a from-scratch developmental learner."""

__all__ = ["AletheiaAgent", "load_config"]
__version__ = "0.34.0"


def __getattr__(name):
    if name == "AletheiaAgent":
        from .agent import AletheiaAgent
        return AletheiaAgent
    if name == "load_config":
        from .config import load_config
        return load_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
