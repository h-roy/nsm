import jax.numpy as jnp
import numpy as np
import optax

from nuox import make_project_grad
from nuox.linalg import (
    dense_solve_lu,
    lstsq_custom_vjp,
    lstsq_lsmr,
    lstsq_via_normaleq,
    solve_iterative_cg,
    solve_materialize,
)


def _constraint_fn(params, matrix):
    return matrix @ params


def test_projection_nullspace_gamma_zero():
    matrix = jnp.array([[1.0, 2.0, -1.0], [0.0, 1.0, 1.0]], dtype=jnp.float32)
    params = jnp.array([0.5, -1.0, 2.0], dtype=jnp.float32)
    grads = jnp.array([1.0, -0.5, 0.25], dtype=jnp.float32)

    solver = lstsq_custom_vjp(lstsq_lsmr())
    transform = make_project_grad(
        _constraint_fn,
        wm_epochs=0,
        num_batches=1,
        gamma=0.0,
        least_squares_solver=solver,
    )
    state = transform.init(params)

    # First step: warmup branch (no projection).
    first_updates, state = transform.update(grads, state, (params, (matrix,), {}))
    np.testing.assert_allclose(np.asarray(first_updates), np.asarray(grads))

    # Second step: projection active.
    projected, _ = transform.update(grads, state, (params, (matrix,), {}))
    residual = matrix @ projected
    assert jnp.linalg.norm(residual) < 1e-5


def test_projection_adds_correction_term():
    matrix = jnp.array([[2.0, 0.0], [1.0, 1.0]], dtype=jnp.float32)
    params = jnp.array([0.3, -0.7], dtype=jnp.float32)
    grads = jnp.array([1.2, -0.4], dtype=jnp.float32)
    gamma = 0.4

    solver = lstsq_custom_vjp(lstsq_lsmr())
    transform = make_project_grad(
        _constraint_fn,
        wm_epochs=0,
        num_batches=1,
        gamma=gamma,
        least_squares_solver=solver,
    )
    state = transform.init(params)
    _, state = transform.update(grads, state, (params, (matrix,), {}))
    projected, _ = transform.update(grads, state, (params, (matrix,), {}))

    constraint = matrix @ params
    correction = matrix @ projected
    np.testing.assert_allclose(
        np.asarray(correction),
        np.asarray(gamma * constraint),
        rtol=1e-4,
        atol=1e-4,
    )


def test_warmup_respected():
    matrix = jnp.eye(3, dtype=jnp.float32)
    params = jnp.array([0.2, 0.1, -0.4], dtype=jnp.float32)
    grads = jnp.array([-0.5, 0.3, 0.8], dtype=jnp.float32)

    solver = lstsq_via_normaleq(solve_iterative_cg(maxiter=200, tol=1e-6))
    transform = make_project_grad(
        _constraint_fn,
        wm_epochs=1,
        num_batches=2,
        gamma=0.0,
        least_squares_solver=solver,
    )
    state = transform.init(params)

    for step in range(3):
        updates, state = transform.update(grads, state, (params, (matrix,), {}))
        if step < 3:
            np.testing.assert_allclose(np.asarray(updates), np.asarray(grads))

    projected, _ = transform.update(grads, state, (params, (matrix,), {}))
    assert jnp.linalg.norm(matrix @ projected) < 1e-5


def test_solver_variants_agree():
    matrix = jnp.array([[1.0, -1.0], [2.0, 0.5]], dtype=jnp.float32)
    params = jnp.array([0.2, -0.3], dtype=jnp.float32)
    grads = jnp.array([0.7, -0.4], dtype=jnp.float32)

    solvers = [
        lstsq_custom_vjp(lstsq_lsmr()),
        lstsq_via_normaleq(solve_iterative_cg(maxiter=500, tol=1e-6)),
        lstsq_via_normaleq(solve_materialize(dense_solve_lu())),
    ]

    projected_updates = []
    for solver in solvers:
        transform = make_project_grad(
            _constraint_fn,
            wm_epochs=0,
            num_batches=1,
            gamma=0.0,
            least_squares_solver=solver,
        )
        state = transform.init(params)
        _, state = transform.update(grads, state, (params, (matrix,), {}))
        projected, _ = transform.update(grads, state, (params, (matrix,), {}))
        projected_updates.append(projected)

    baseline = projected_updates[0]
    for upd in projected_updates[1:]:
        np.testing.assert_allclose(
            np.asarray(upd),
            np.asarray(baseline),
            rtol=1e-5,
            atol=1e-5,
        )
