# NSM: Null-space optimisation for JAX

[![Actions status](https://github.com/h-roy/deep-nsm/workflows/ci/badge.svg)](https://github.com/h-roy/deep-nsm/actions)
[![PyPI version](https://img.shields.io/pypi/v/nsm.svg)](https://pypi.org/project/nsm/)
[![PyPI license](https://img.shields.io/pypi/l/nsm.svg)](https://pypi.org/project/nsm/)
[![Python versions](https://img.shields.io/pypi/pyversions/nsm.svg)](https://pypi.org/project/nsm/)

Null-space Method (NSM) provides Optax-compatible gradient transformations that enforce equality
constraints through differentiable least-squares solves. The package extracts and hardens the
projection code from the `code-projected-constraints` research project.

- ⚡ Projection transforms for Optax optimisers that keep constraint Jacobians satisfied
- ⚡ Differentiable least-squares solvers (LSMR, normal equations) for constraint systems
- ⚡ Works with PyTrees, JIT compilation, and automatic differentiation
- ⚡ Designed for constrained training loops in JAX-based deep-learning workloads
- ⚡ Documentation powered by MkDocs + mkdocstrings

[_Let us know what you build with NSM!_](https://github.com/h-roy/deep-nsm/issues)


**Installation**

```commandline
pip install nsm
```

**Important:** NSM assumes you already installed JAX. Follow the
[official instructions](https://github.com/google/jax#installation) for the correct wheel. For
development you can grab all tooling dependencies at once:

```commandline
pip install -e .[dev]
```

This installs pytest, numpy, and linting utilities.
To work on the documentation run

```commandline
pip install -e .[docs]
```


**Minimal example**

```python
import optax
import jax
import jax.numpy as jnp

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
```


**Tutorials**

- Walkthroughs and API docs: <https://h-roy.github.io/deep-nsm/>
- Example notebooks from the original research repo:
  <https://github.com/h-roy/code-projected-constraints>
- Lightweight demos inside `examples/`:
  [TODO]

**Citation**

If NSM contributes to your research, please cite the projection-derivatives paper when appropriate. [TODO: Add arxiv link]


## Develop with NSM

Install all test dependencies (JAX must already be set up for your accelerator):

```commandline
pip install -e .[dev]
```

Quick checks:

```commandline
pytest                      # run the unit tests
```

Serve the documentation locally:

```commandline
pip install -e .[docs]
mkdocs serve
```

## Contribute

Contributions are very welcome! Please:

1. Open an issue if you plan a new feature or find a bug.
2. Fork the repository and install dependencies with `pip install -e .[dev]`.
3. Add tests or examples demonstrating the change.
4. Run `pytest` (and optionally `ruff check` if you install the formatter).
5. Submit a pull request with a clear title and link the related issue.

Feel free to propose additional tutorials, solvers, or integrations that make constrained training easier for the community.
