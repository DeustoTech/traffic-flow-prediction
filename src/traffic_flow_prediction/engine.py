import jax
import jax.numpy as jnp
import jraph
import optax
from flax import nnx
from jax.typing import ArrayLike


def weight_decay(t_max: int, decay: float = 0.05) -> jax.Array:
    t = jnp.arange(t_max)
    return jnp.exp(-decay * t)


@nnx.jit
def train_step(
    model:     nnx.Module,
    optimiser: nnx.Optimizer,
    metric:    nnx.Metric,
    x:         jraph.GraphsTuple,
    y:         ArrayLike,
    *,
    decay: float = 0.05,
    alpha: float = 0.9,
) -> jax.Array:
    y_diff = jnp.diff(y, axis=1)
    
    def loss_fn(model: nnx.Module) -> jax.Array:
        y_pred = model(x)
        y_pred_diff = jnp.diff(y_pred, axis=1)
        w = weight_decay(y.shape[1], decay).reshape(1, -1, 1)
        loss1 = jnp.mean(jnp.sum(w * optax.losses.squared_error(y_pred, y), axis=1))
        loss2 = jnp.mean(jnp.sum(w[:, :-1, :] * optax.losses.squared_error(y_pred_diff, y_diff), axis=1))
        return alpha * loss1 + (1 - alpha) * loss2
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimiser.update(grads)
    metric.update(loss=loss)
    return loss


@nnx.jit
def evaluate(
    model:   nnx.Module,
    metrics: nnx.MultiMetric,
    x:       jraph.GraphsTuple,
    y:       ArrayLike,
) -> jax.Array:
    y = y[:, 0, :]
    y_pred = model(x)[:, 0, :]
    mse = jnp.mean(optax.losses.squared_error(y_pred, y))
    mae = jnp.mean(jnp.abs(y_pred - y))
    metrics.update(mse=mse, mae=mae)
    return y_pred
