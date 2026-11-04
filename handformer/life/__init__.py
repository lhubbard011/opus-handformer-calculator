"""Conway's Game of Life, written by hand into transformer weights (see circuit.py, LIFE.md)."""
from .circuit import ALLOWED, DIMS, OUTPUTS, build, builder, check, prompt, solve

__all__ = ["ALLOWED", "DIMS", "OUTPUTS", "build", "builder", "check", "prompt", "solve"]
