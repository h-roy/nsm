import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import linen as nn
from jax import nn as jnn


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


def make_data(num_points: int, *, key: jax.Array) -> jnp.ndarray:
    return jnp.linspace(-jnp.pi, jnp.pi, num_points, dtype=jnp.float32).reshape(-1, 1)


key = jax.random.PRNGKey(0)
x_train = make_data(256, key=key)
y_train = jnp.sin(x_train)
model = MLP(output_dim=3, hidden_dim=64, num_layers=3, activation="tanh")
model_key, key = jax.random.split(key)
model_init, model_apply_ = model.init, model.apply
params = model_init(model_key, x_train.squeeze())
breakpoint()
