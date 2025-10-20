"""Quick comparison of NSM linear solvers.

Run with `PYTHONPATH=src python tmp_linear_solver.py`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nsm.linalg import (
    dense_solve_lu,
    lstsq_lsmr,
    lstsq_via_normaleq,
    solve_iterative_cg,
    solve_materialize,
)


def run_single(shape=(8, 5), seed: int = 0) -> None:
    key = jax.random.PRNGKey(seed)
    A = jax.random.normal(key, shape, dtype=jnp.float32)
    subkey, _ = jax.random.split(key)
    x_true = jax.random.normal(subkey, (shape[1],), dtype=jnp.float32)
    b = A @ x_true

    numpy_sol = jnp.linalg.lstsq(A, b)[0]

    matrix_T = A.T

    def vecmat(v, *_):
        return matrix_T @ v

    solvers = [
        ("LSMR", lstsq_lsmr()),
        ("NormalEq+CG", lstsq_via_normaleq(solve_iterative_cg(maxiter=5_000, tol=1e-6))),
        ("NormalEq+LU", lstsq_via_normaleq(solve_materialize(dense_solve_lu()))),
    ]

    print(f"Problem shape A: {shape}, seed: {seed}")
    print("solver          rel_error    residual")
    for name, solver in solvers:
        solution, _stats = solver(vecmat, b, vecmat_args=())
        rel_error = jnp.linalg.norm(solution - numpy_sol) / jnp.linalg.norm(numpy_sol)
        residual = jnp.linalg.norm(A @ solution - b)
        print(f"{name:15s} {float(rel_error):.3e}   {float(residual):.3e}")


if __name__ == "__main__":
    run_single((8, 5), seed=0)
    print()
    run_single((5, 8), seed=1)
    print()
    run_single((6, 6), seed=2)
