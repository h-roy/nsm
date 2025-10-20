"""Differentiable null-space optimizers for Optax."""

from .operators import build_constraint_ops, vectorize_model
from .optax_nullspace import (
    ProjState,
    make_project_grad,
    make_update_fn,
)

__all__ = [
    "ProjState",
    "make_project_grad",
    "make_update_fn",
    "build_constraint_ops",
    "vectorize_model",
]

__version__ = "0.1.0"
