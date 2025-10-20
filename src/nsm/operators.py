from __future__ import annotations

from typing import Any, Callable, Sequence, Tuple

import jax
from flax import linen as nn
from flax import nnx
from flax.core import FrozenDict
from jax.flatten_util import ravel_pytree

ConstraintFn = Callable[..., Any]
LinearOp = Tuple[
    Callable[[Any, Any], Any],
    Callable[[Any, Any], Any],
]


def build_constraint_ops(constraint_fn: ConstraintFn) -> Callable[..., LinearOp]:
    """Return matvec and rmatvec closures for the constraint Jacobian.

    The returned callable expects positional ``model_args`` and keyword constraint
    arguments. When invoked with concrete values, it produces two functions:

    ``matvec(v, params)`` applies the Jacobian of ``constraint_fn`` to ``v``.
    ``vecmat(v, params)`` applies the transpose Jacobian to ``v``.
    """

    def make_ops(*model_args, **constraint_kwargs):
        def constrained_params_fn(p):
            return constraint_fn(p, *model_args, **constraint_kwargs)

        def matvec(v, params):
            return jax.jvp(constrained_params_fn, (params,), (v,))[1]

        def vecmat(v, params):
            _, vjp_fn = jax.vjp(constrained_params_fn, params)
            return vjp_fn(v)[0]

        return matvec, vecmat

    return make_ops


def vectorize_model(
    model: Any,
    *,
    params: Any | None = None,
    variables: Any | None = None,
    mutable: Any | None = None,
    nnx_filter: Any | None = None,
    nnx_state_filters: Sequence[Any] = (),
) -> Tuple[Callable[..., Any], jax.Array, Callable[[jax.Array], Any]]:
    if isinstance(model, nn.Module):
        return _vectorize_linen(model, params, variables=variables, mutable=mutable)
    else:
        return _vectorize_nnx(model, nnx_filter, nnx_state_filters)


def _vectorize_linen(
    module: nn.Module,
    params: Any,
    *,
    variables: Any | None = None,
    mutable: Any | None = None,
) -> Tuple[Callable[..., Any], jax.Array, Callable[[jax.Array], Any]]:
    params_flat, unravel = ravel_pytree(params)

    if variables is None:
        base_variables: dict[str, Any] = {}
    elif isinstance(variables, FrozenDict):
        base_variables = dict(variables)
    else:
        base_variables = dict(variables)

    def apply_vec(p_vec, *args, **kwargs):
        params_tree = unravel(p_vec)
        vars_all = dict(base_variables)
        vars_all["params"] = params_tree

        if mutable:
            outputs = module.apply(vars_all, *args, **kwargs, mutable=mutable)
            if isinstance(outputs, tuple):
                return outputs[0]
            return outputs
        return module.apply(vars_all, *args, **kwargs)

    return apply_vec, params_flat, unravel


def _vectorize_nnx(
    module: nnx.Module,
    nnx_filter: Any | None,
    nnx_state_filters: Sequence[Any],
) -> Tuple[Callable[..., Any], jax.Array, Callable[[jax.Array], Any]]:
    filter_param = nnx.Param if nnx_filter is None else nnx_filter
    split_items = nnx.split(module, filter_param, *nnx_state_filters)
    graphdef = split_items[0]
    params = split_items[1]
    other_states = tuple(nnx.pure(s) for s in split_items[2:])

    params_flat, unravel = ravel_pytree(params)

    def apply_vec(p_vec, *args, **kwargs):
        params_tree = unravel(p_vec)
        call = graphdef.apply(params_tree, *other_states)
        outputs = call(*args, **kwargs)
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs

    return apply_vec, params_flat, unravel
