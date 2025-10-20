import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import flatten_util

from nuox.operators import vectorize_model

# Test that vectorizing nns give the same result clean up all the flax shit


@pytest.mark.parametrize("features", [2, 4])
def test_vectorize_linen_dense(features):
    flax = pytest.importorskip("flax")
    try:
        from flax import linen as nn
    except ImportError:
        pytest.skip("Flax installation lacks linen module")

    class DenseModel(nn.Module):
        out_features: int

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(self.out_features)(x)
            return jax.nn.tanh(x)

    model = DenseModel(features)
    key = jax.random.PRNGKey(0)
    x = jnp.ones((3, 5))
    variables = model.init(key, x)

    apply_vec, params_vec, unravel = vectorize_model(model, params=variables["params"])

    out_ref = model.apply(variables, x)
    out_vec = apply_vec(params_vec, x)

    np.testing.assert_allclose(np.asarray(out_vec), np.asarray(out_ref), rtol=1e-6, atol=1e-6)

    def loss_flat(flat_params):
        preds = apply_vec(flat_params, x)
        return jnp.sum(preds**2)

    flat_grad = jax.grad(loss_flat)(params_vec)

    def loss_tree(p):
        preds = model.apply({"params": p}, x)
        return jnp.sum(preds**2)

    tree_grad = jax.grad(loss_tree)(variables["params"])
    tree_grad_flat, _ = flatten_util.ravel_pytree(tree_grad)

    np.testing.assert_allclose(
        np.asarray(flat_grad), np.asarray(tree_grad_flat), rtol=1e-5, atol=1e-5
    )


def test_vectorize_nnx_linear():
    flax = pytest.importorskip("flax")
    try:
        from flax import nnx
    except ImportError:
        pytest.skip("Flax installation lacks nnx module")

    class Net(nnx.Module):
        def __init__(self, *, rngs):
            self.linear = nnx.Linear(5, 3, rngs=rngs)

        def __call__(self, x):
            return jax.nn.relu(self.linear(x))

    rngs = nnx.Rngs(0)
    model = Net(rngs=rngs)
    x = jnp.ones((4, 5))

    apply_vec, params_vec, _ = vectorize_model(model)

    out_ref = model(x)
    out_vec = apply_vec(params_vec, x)

    np.testing.assert_allclose(np.asarray(out_vec), np.asarray(out_ref), rtol=1e-6, atol=1e-6)
