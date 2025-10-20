from __future__ import annotations

from typing import Any, Callable, Dict, NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp
import optax

from .operators import build_constraint_ops

PyTree = Any
ConstraintFn = Callable[..., jax.Array]
LeastSquaresSolver = Callable[..., Tuple[PyTree, Dict[str, Any]]]


class ProjState(NamedTuple):
    """State for the projection transform."""

    count: chex.Array


def make_project_grad(
    constraint_fn: ConstraintFn,
    *,
    wm_epochs: int,
    num_batches: int,
    gamma: float,
    least_squares_solver: LeastSquaresSolver,
    scale_gamma: bool = False,
) -> optax.GradientTransformation:
    """Create a null-space projection transform for Optax.

    Args:
        constraint_fn: Callable returning the constraint violation for a given set
            of parameters. Signature: ``constraint_fn(params, *model_args, **kwargs)``.
        wm_epochs: Warmup epochs where the projection is skipped.
        num_batches: Number of batches per epoch used to compute warmup steps.
        gamma: Rate used for the Gauss-Newton correction term.
        least_squares_solver: Callable implementing the least-squares solve.
            Must accept ``(vecmat, rhs, vecmat_args=())`` and return
            ``(solution, stats)``.
        scale_gamma: If ``True``, scale ``gamma`` by the constraint norm.
    """

    ops_builder = build_constraint_ops(constraint_fn)

    def lstsq_grad(grads, params, *model_args, **constraint_kwargs):
        matvec, vecmat = ops_builder(*model_args, **constraint_kwargs)
        constraint = constraint_fn(params, *model_args, **constraint_kwargs)

        if scale_gamma:
            scaled_gamma = gamma * jnp.linalg.norm(constraint)
        else:
            scaled_gamma = gamma

        rhs = matvec(grads, params) - scaled_gamma * constraint
        lstsq_grads, _ = least_squares_solver(vecmat, rhs, vecmat_args=(params,))
        return grads - lstsq_grads

    def init_fn(_params):
        return ProjState(count=jnp.zeros([], dtype=jnp.int32))

    def update_fn(updates, state, update_params):
        params, model_args, constraint_kwargs = update_params
        num_steps = state.count
        project = lambda u: lstsq_grad(u, params, *model_args, **constraint_kwargs)
        updates = jax.lax.cond(
            num_steps > wm_epochs * num_batches,
            project,
            lambda u: u,
            updates,
        )
        next_state = ProjState(count=num_steps + 1)
        return updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)


def make_update_fn(loss_fn, optim, constr_fn):
    """Create an Optax-compatible update function matching the legacy API."""

    @jax.jit
    def update_fn(params, opt_state, x, y, **constr_params):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optim.update(grads, opt_state, (params, (), constr_params))
        params = optax.apply_updates(params, updates)
        constraint_value = constr_fn(params, **constr_params)
        return loss, params, opt_state, constraint_value

    return update_fn
