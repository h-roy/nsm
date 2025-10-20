import jax
import jax.numpy as jnp
import optax

from nsm import make_project_grad
from nsm.linalg import lstsq_custom_vjp, lstsq_lsmr


def constraint_fn(params, matrix):
    # Keep parameters on an affine hyperplane.
    return matrix @ params - 1.0


normal = jnp.array([1.0, -2.0, 1.0])
solver = lstsq_custom_vjp(lstsq_lsmr())
transform = make_project_grad(
    constraint_fn,
    wm_epochs=0,
    num_batches=1,
    gamma=0.1,
    least_squares_solver=solver,
)
optimizer = optax.chain(transform, optax.adam(1e-3))


@jax.jit
def step(opt_state, params, batch):
    def loss_fn(p):
        pred = jnp.dot(p, batch["features"])
        return 0.5 * jnp.square(pred - batch["targets"]).mean()

    grads = jax.grad(loss_fn)(params)
    updates, opt_state = optimizer.update(
        grads,
        opt_state,
        (params, (normal,), {}),
    )
    params = optax.apply_updates(params, updates)
    return opt_state, params
