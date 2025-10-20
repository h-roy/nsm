# NUOX: Null-space Optimization for Neural Networks

[![PyPI version](https://badge.fury.io/py/nuox.svg)](https://badge.fury.io/py/nuox)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/arXiv-2106.0128-b31b1b.svg)](https://your-paper-link.com)

**`nuox`** is a JAX-based library that provides powerful, scalable, and differentiable tools for structured machine learning problems. It integrates state-of-the-art adaptive least-squares solvers (like LSMR) into modern deep learning workflows.

Welcome to the official documentation for **`nuox`**, a JAX-based Python library for imposing complex constraints on neural networks.

This package provides a simple and powerful way to solve structured learning problems by implementing the **Null-Space Method**. Instead of relying on hand-tuned penalty terms or complex custom architectures, you can define almost any constraint as a simple function and use a standard optimizers like Adam to solve it.

***

## Installation

You can install **`nuox`** directly from PyPI:

```bash
pip install nuox
```

### The Core Idea: The Null-Space Method

Many important structural properties—like equivariance, sparsity, or conservativeness—can be expressed as a constraint, $c(\theta) = 0$, on a model's parameters. However, standard optimizers like SGD or Adam are designed for unconstrained problems and struggle to enforce such conditions.

The Null-Space Method solves this by cleverly transforming the gradient at each training step. It takes the proposed update from a standard optimizer (e.g., an Adam step) and projects it onto the null-space of the *linearized* constraint. This ensures that the optimization step makes progress on the task loss while simultaneously keeping the constraint locally constant. An additional, constraint correction term is added to the update ensuring that upon convergence the model satisfies the constraint. This is how null-space method differs from the simple approach of optimizing a penalized loss, since for finite penalty weights constraint satisfaction is not guaranteed in the latter.

The best part? This entire projection is handled by a fast, differentiable least-squares solvers, and our library wraps it in a simple API.

***

### How to Use **`nuox`**

With **`nuox`**, you can turn any [Optax](https://optax.readthedocs.io/en/latest/) optimizer into a constrained optimizer in just three steps, as shown in your repository's code.

#### **Step 1: Define Your Constraint as a Function**

Create a Python function that takes your model's parameters and returns the current value of your constraint vector, $c(\theta)$. This function is the heart of your structured learning task.

    # Import the core components from your library
    from nuox import make_project_grad
    from nuox.linalg import lstsq_custom_vjp, lstsq_lsmr

    # This function must return a JAX array representing the constraint violation.
    def constraint_fn(params, *model_args, **constraint_kwargs):
        matrix = model_args[0]
        return matrix @ params - constraint_kwargs["target"]

#### **Step 2: Create the Null-Space Projection Transform**

Use `make_project_grad` to create a transformation that enforces your constraint. You provide the constraint function and a least-squares solver (for instance, LSMR).

    # Wrap LSMR with a differentiable interface and use it in the projection
    solver = lstsq_custom_vjp(lstsq_lsmr(atol=1e-6, btol=1e-6))
    transform = make_project_grad(
        constraint_fn=constraint_fn,
        least_squares_solver=solver,
        wm_epochs=0,
        num_batches=1,
        gamma=0.1,
    )

#### **Step 3: Chain it with any Optax Optimizer**

Finally, use `optax.chain` to combine the null-space projection with a standard optimizer. The projection will be applied first, ensuring every update respects the constraint.

    import optax

    # Now, `optimizer` is a constrained optimizer!
    optimizer = optax.chain(
        transform,
        optax.adam(learning_rate=1e-3)
    )

    # Each update receives (params, model_args, constraint_kwargs)
    updates, state = optimizer.update(
        grads,
        state,
        (params, (matrix,), {"target": 0.0})
    )

You can now use this `optimizer` in your training loop just like any other Optax optimizer. Your model will be optimized for the task loss while simultaneously satisfying the specified structure.

***

### Citing this Work

If you use **`nuox`** in your research, please cite the following paper:

```
@article{anonymous2025structured,
  title={Structured learning with adaptive least squares},
  author={Anonymous Author(s)},
  journal={Submitted to NeurIPS},
  year={2025}
}
```
