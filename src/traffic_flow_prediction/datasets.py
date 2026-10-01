from collections.abc import Iterator

import jax
import jax.numpy as jnp
import jax.random as jrand
import pandas as pd
from flax import nnx
from jax.typing import ArrayLike


class DataLoader:
    
    def __init__(self, x: ArrayLike, y: ArrayLike, batch_size: int, shuffle: bool = False, rngs: nnx.Rngs | None = None) -> None:
        x = jnp.asarray(x)
        y = jnp.asarray(y)

        if len(x) != len(y):
            raise ValueError("x and y must have the same number of samples.")

        self._x = x
        self._y = y
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._rngs = rngs

    def __iter__(self) -> Iterator[tuple[jax.Array, jax.Array]]:
        n = len(self._x)

        if self._shuffle:
            if self._rngs is None:
                raise ValueError("A PRNG key must be provided when shuffle=True.")
            indices = jrand.permutation(self._rngs(), n)
        else:
            indices = jnp.arange(n)

        for start in range(0, n, self._batch_size):
            batch_idx = indices[start : start + self._batch_size]
            yield self._x[batch_idx], self._y[batch_idx]

    def __len__(self) -> int:
        return (len(self._x) + self._batch_size - 1) // self._batch_size


def to_edges(df_edges: pd.DataFrame) -> tuple[jax.Array, jax.Array, jax.Array]:
    if df_edges.empty:
        raise ValueError("df_edges")
    m = len(df_edges)
    if "weight" in df_edges.columns:
        edges = jnp.asarray(df_edges["weight"].values, dtype=jnp.float32)[:, jnp.newaxis]
    else:
        edges = jnp.ones((m, 1), dtype=jnp.float32)
    receivers = jnp.asarray(df_edges["target"].values, dtype=jnp.int32)
    senders   = jnp.asarray(df_edges["source"].values, dtype=jnp.int32)
    return edges, receivers, senders


def create_windows(data: ArrayLike, window: int = 32, horizon: int = 1) -> tuple[jax.Array, jax.Array]:
    x = []
    y = []
    n_time = data.shape[0]
    for t in range(n_time - window - horizon + 1):
        x_t = data[t          : t + window]
        y_t = data[t + window : t + window + horizon]
        x.append(x_t[..., jnp.newaxis])
        y.append(y_t[..., jnp.newaxis])
    if not x:
        raise ValueError(f"There is no sufficient time iteration ({n_time}) for windows={window} and horizon={horizon}.")
    return jnp.asarray(x, dtype=jnp.float32), jnp.asarray(y, dtype=jnp.float32)


def split_time(
    x: ArrayLike,
    train_ratio: float = 0.70
) -> tuple[jax.Array, jax.Array, jax.Array]:
    n = len(x)
    n_train = int(train_ratio * n)
    n_valid = (n - n_train) // 2
    return (
        x[                  : n_train          ],
        x[n_train           : n_train + n_valid],
        x[n_train + n_valid :                  ],
    )


def _split_time(
    x: ArrayLike,
    y: ArrayLike,
    train_ratio: float = 0.70
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    n = len(x)
    n_train = int(train_ratio * n)
    n_valid = (n - n_train) // 2
    return (
        x[                  : n_train          ], y[                  : n_train          ],
        x[n_train           : n_train + n_valid], y[n_train           : n_train + n_valid],
        x[n_train + n_valid :                  ], y[n_train + n_valid :                  ],
    )
