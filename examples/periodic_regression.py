import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import linen as nn
from jax import nn as jnn

from nsm import make_project_grad
from nsm.linalg import lstsq_custom_vjp, lstsq_lsmr
from nsm.optax_nullspace import make_update_fn

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_cublaslt=true --xla_gpu_cublas_fallback=true --xla_gpu_enable_command_buffer=''"
)


class MLP(nn.Module):
    output_dim: 1
    hidden_dim: 64
    num_layers: 3
    activation: "tanh"

    def act_fun(self, x):
        if self.activation == "tanh":
            return jnn.tanh(x)
        if self.activation == "relu":
            return jnn.relu(x)

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = self.act_fun(x)
        x = nn.Dense(self.output_dim)(x)
        return x


def train_baseline(params, predict, x, y, *, steps: int, lr: float):
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @jax.jit
    def step(theta, state):
        def loss_fn(t):
            preds = predict(t, x)
            return 0.5 * jnp.mean((preds - y) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(theta)
        updates, state = opt.update(grads, state)
        theta = optax.apply_updates(theta, updates)
        return theta, state, loss

    losses = []
    for iter in range(steps):
        params, opt_state, loss = step(params, opt_state)
        losses.append(loss)
        print(f"step={iter + 1}, mse_loss={loss:.3f}")

    return params, jnp.stack(losses)


def train_constrained(
    params,
    predict,
    x,
    y,
    collocation,
    *,
    steps: int,
    lr: float,
    gamma: float,
    wm_epochs: int,
):
    def loss_fn(theta, inputs, targets):
        preds = predict(theta, inputs)
        return 0.5 * jnp.mean((preds - targets) ** 2)

    def constraint_fn(theta, *, points, period):
        preds = predict(theta, points)
        preds_shifted = predict(theta, points + period)
        return preds_shifted - preds

    solver = lstsq_custom_vjp(lstsq_lsmr(maxiter=10))
    transform = make_project_grad(
        constraint_fn,
        wm_epochs=wm_epochs,
        num_batches=1,
        gamma=gamma,
        least_squares_solver=solver,
    )
    optim = optax.chain(transform, optax.adam(lr))
    update_fn = make_update_fn(loss_fn, optim, constraint_fn)
    opt_state = optim.init(params)

    losses = []
    residuals = []
    for iter in range(steps):
        loss, params, opt_state, constr = update_fn(
            params,
            opt_state,
            x,
            y,
            points=collocation,
            period=2.0 * jnp.pi,
        )
        losses.append(loss)
        residuals.append(jnp.linalg.norm(constr) / constr.size)
        print(f"step={iter + 1}, mse_loss={loss:.3f}, constr={jnp.linalg.norm(constr):.3f}")

    return params, jnp.stack(losses), jnp.stack(residuals)


def make_data(num_points):
    return jnp.linspace(-2 * jnp.pi, 2 * jnp.pi, num_points).reshape(-1, 1)


def main():
    key = jax.random.PRNGKey(0)
    x_train = make_data(500)
    collocation = jax.random.normal(key, (200,)).reshape(-1, 1) * 4 - 2 * jnp.pi
    y_train = jnp.sin(x_train)
    model = MLP(output_dim=1, hidden_dim=16, num_layers=2, activation="tanh")
    model_key, key = jax.random.split(key)
    model_init, model_apply_ = model.init, model.apply
    params = model_init(model_key, x_train)
    params, unflatten = jax.flatten_util.ravel_pytree(params)
    D = len(params)
    predict = lambda p, x: model_apply_(unflatten(p), x)

    baseline_params, baseline_losses = train_baseline(
        params,
        predict,
        x_train,
        y_train,
        steps=500,
        lr=3e-3,
    )
    constrained_params, constrained_losses, residuals = train_constrained(
        params,
        predict,
        x_train,
        y_train,
        collocation,
        steps=5000,
        lr=3e-3,
        gamma=0.1,
        wm_epochs=20,
    )
    x_plot = jnp.linspace(-3 * jnp.pi, 3 * jnp.pi, 1000).reshape(-1, 1)
    y_true = jnp.sin(x_plot).squeeze(-1)
    baseline_pred = predict(baseline_params, x_plot)
    constrained_pred = predict(constrained_params, x_plot)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    axes[0].plot(x_plot, y_true, label="Target", color="black", linewidth=2)
    axes[0].plot(x_plot, baseline_pred, label="Baseline", color="#E07A5F")
    axes[0].plot(x_plot, constrained_pred, label="NSM", color="#3D405B")
    axes[0].set_ylabel("f(x)")
    axes[0].legend(loc="upper right")

    axes[1].plot(baseline_losses, label="Baseline loss", color="#E07A5F")
    axes[1].plot(constrained_losses, label="NSM loss", color="#3D405B")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].twinx().plot(residuals, label="NSM residual", color="#2A9D8F")
    axes[1].legend(loc="upper right")

    path = "examples/periodic_regression.pdf"
    fig.savefig(path, dpi=200)
    print("Saved figure to", path)


if __name__ == "__main__":
    main()
