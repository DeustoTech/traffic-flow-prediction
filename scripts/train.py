import argparse
import sys
import time
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

sys.path.append(Path("src").absolute().as_posix())

import jax
import jax.numpy as jnp
import jax.tree_util as jtree
import jraph
import matplotlib.pyplot as plt
import optax
from flax import nnx
from jax.typing import ArrayLike
from tensorboardX import SummaryWriter

from traffic_flow_prediction.datasets import DataLoader
from traffic_flow_prediction.engine import evaluate, train_step
from traffic_flow_prediction.models import GNN, SpatialType, TemporalType

fprint = partial(print, flush=True)


def fprint_tree(tree: Any, name: str = "tree", shape: bool = True) -> None:
    if shape:
        jtree.tree_map_with_path(lambda p, v: fprint(f"{name}{jtree.keystr(p)}.shape = {v.shape}"), tree)
    else:
        jtree.tree_map_with_path(lambda p, v: fprint(f"{name}{jtree.keystr(p)}\n{v}"), tree)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",          type=int,            default=42)
    parser.add_argument("--source",        type=str,            default="data/processed/base/closeness")
    parser.add_argument("--target",        type=str,            default="target")
    parser.add_argument("--nodes",         type=int, nargs="+", default=())
    parser.add_argument("--tensorboard",   action="store_true", default=False)
    
    data = parser.add_argument_group()
    data.add_argument("--batch-size", type=int,            default=32)
    data.add_argument("--shuffle",    action="store_true", default=False)
    
    model = parser.add_argument_group()
    model.add_argument("--hidden-feats", type=int, default=64)
    model.add_argument("--layers",       type=int, default=2)
    model.add_argument("--spatial",      type=str, default=SpatialType.GCN.name,   choices=[x.name for x in SpatialType])
    model.add_argument("--temporal",     type=str, default=TemporalType.LSTM.name, choices=[x.name for x in TemporalType])
    model.add_argument("--autoregressive", action="store_true", default=False)
    
    optim = parser.add_argument_group()
    optim.add_argument("--epochs",        type=int,   default=10)
    optim.add_argument("--learning-rate", type=float, default=1e-4)
    optim.add_argument("--time-decay",    type=float, default=0.05)
    optim.add_argument("--alpha",         type=float, default=0.9)
    
    return parser.parse_args()


def load_data(path_to_source: Path) -> tuple[dict[str, ArrayLike], dict[str, ArrayLike], dict[str, ArrayLike], dict[str, ArrayLike]]:
    graph_npz = jnp.load(path_to_source / "graph.npz", allow_pickle=True)
    train_npz = jnp.load(path_to_source / "train.npz", allow_pickle=True)
    valid_npz = jnp.load(path_to_source / "valid.npz", allow_pickle=True)
    test_npz  = jnp.load(path_to_source / "test.npz",  allow_pickle=True)
    graph_data = {k: graph_npz[k] for k in graph_npz}
    train_data = {k: train_npz[k] for k in train_npz}
    valid_data = {k: valid_npz[k] for k in valid_npz}
    test_data  = {k: test_npz[k]  for k in test_npz}
    return graph_data, train_data, valid_data, test_data


def prepare_data(x: ArrayLike, y: ArrayLike, graph_data: Mapping[str, ArrayLike]) -> tuple[jraph.GraphsTuple, jax.Array]:
    batch_dim, time_dim, _, feat_dim = x.shape
    nodes_idx = graph_data["nodes_idx"]
    
    n = nodes_idx.max().item() + 1
    nodes = jnp.zeros((batch_dim, time_dim, n, feat_dim), dtype=x.dtype)
    nodes = nodes.at[:, :, nodes_idx, :].set(x)
    nodes = jnp.swapaxes(nodes, 1, 2)
    nodes = jnp.concatenate(nodes, axis=0)
    
    m = graph_data["edges"].shape[0]
    edges = graph_data["edges"]
    edges = jnp.concatenate((edges) * batch_dim)
    
    x_graph = jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        receivers=graph_data["receivers"],
        senders=graph_data["senders"],
        globals=None,
        n_node=jnp.full((batch_dim,), n, dtype=jnp.int32),
        n_edge=jnp.full((batch_dim,), m, dtype=jnp.int32),
    )
    
    batch_dim, time_dim, _, feat_dim = y.shape
    y_array = jnp.zeros((batch_dim, time_dim, n, feat_dim), dtype=y.dtype)
    y_array = y_array.at[:, :, nodes_idx, :].set(y)
    y_array = jnp.swapaxes(y_array, 1, 2)
    y_array = jnp.concatenate(y_array, axis=0)
    
    return x_graph, y_array


@jax.jit
def get_signal(x: jraph.GraphsTuple, y: ArrayLike) -> jax.Array:
    n_graph = x.n_node.shape[0]
    ys = jnp.array_split(y[:, 0], n_graph, axis=0)
    return jnp.stack(ys, axis=1)
    # n_batch = x.n_node.shape[0]
    # n_sensor = y.shape[0] // n_batch
    # n_time = y.shape[1]
    # ys = jnp.array_split(y, n_batch, axis=0)
    # ys = jtree.tree_map(
    #     lambda yi, i: jnp.zeros((n_sensor, n_batch + n_time - 1)).at[:, i:(i+n_time)].set(yi),
    #     ys,
    #     list(range(n_batch)),
    # )
    # z = jnp.stack(ys, axis=0)
    
    # r = jnp.arange(n_time, dtype=jnp.float32)
    # n = jnp.zeros(n_batch + n_time - 1).at[-n_time:].add(r).at[:n_time].add(jnp.flip(r))
    # n = jnp.reshape(n, (1, -1))
    # return jnp.sum(z, axis=0) / (n_time - n)


def plot_signals(
    path_to_target: Path,
    signal_pred:    ArrayLike,
    signal_targ:    ArrayLike,
    *nodes:         int,
) -> None:
    for node in nodes:
        fig, ax = plt.subplots()
        ax.plot(signal_pred[node], zorder=2, label="prediction")
        ax.plot(signal_targ[node], zorder=1, label="target")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels)
        fig.tight_layout()
        fig.savefig(path_to_target / f"node_{node}")


def log_results_train(
    train_metric: nnx.Metric,
    valid_metric: nnx.MultiMetric,
    step:         int,
    writer:       SummaryWriter | None,
) -> None:
    values = valid_metric.compute()
    fprint(f"Epoch: {step:<3d}" +
           f" | Train loss: {train_metric.compute():<15.10f}" +
           f" | Valid MSE: {values['mse']:<15.10f}" +
           f" | Valid MAE: {values['mae']:<15.10f}" )
    if writer:
        writer.add_scalar("train/loss", train_metric.compute(), step)
        writer.add_scalar("valid/mse",  values["mse"],          step)
        writer.add_scalar("valid/mae",  values["mae"],          step)


def show_results_test(
    path_to_target: Path,
    model:          GNN,
    loader:         DataLoader,
    graph_data:     Mapping[str, ArrayLike],
    *nodes:         int,
) -> None:
    path_to_target.mkdir(parents=True, exist_ok=True)
    metric = nnx.MultiMetric(mse=nnx.metrics.Average("mse"), mae=nnx.metrics.Average("mae"))
    signal_pred = []
    signal_targ = []
    for x_batch, y_batch in loader:
        x, y = prepare_data(x_batch, y_batch, graph_data)
        y_pred = evaluate(model, metric, x, y)
        signal_pred.append(get_signal(x, y_pred))
        signal_targ.append(get_signal(x, y[:, 0, :]))
    signal_pred = jnp.concatenate(signal_pred, axis=1)
    signal_targ = jnp.concatenate(signal_targ, axis=1)
    values = metric.compute()
    fprint("Test MSE:", values["mse"])
    fprint("Test MAE:", values["mae"])
    plot_signals(path_to_target, signal_pred, signal_targ, *nodes)


def main() -> None:
    args = get_args()
    rngs = nnx.Rngs(args.seed, params=args.seed + 1)
    
    path_to_source = Path(args.source)
    path_to_target = Path(args.target, f"exp_{time.strftime('%y%m%d%H%M%S')}")
    path_to_target.mkdir(parents=True, exist_ok=True)
    
    graph_data, train_data, valid_data, test_data = load_data(path_to_source)
    train_loader = DataLoader(train_data["x"], train_data["y"], args.batch_size, args.shuffle, rngs)
    valid_loader = DataLoader(valid_data["x"], valid_data["y"], args.batch_size, args.shuffle, rngs)
    test_loader  = DataLoader( test_data["x"],  test_data["y"], args.batch_size, args.shuffle, rngs)
    
    x_0, y_0 = next(iter(train_loader))
    x, y = prepare_data(x_0, y_0, graph_data)
    _, in_time, in_feat  = x.nodes.shape
    _, out_time, _ = y.shape
    
    model = GNN(
        in_feat,
        args.hidden_feats,
        args.layers,
        in_time,
        out_time,
        spatial=SpatialType[args.spatial],
        temporal=TemporalType[args.temporal],
        rngs=rngs,
    )
    optimiser = nnx.Optimizer(model, optax.adam(args.learning_rate))
    train_metric = nnx.metrics.Average("loss")
    valid_metric = nnx.MultiMetric(mse=nnx.metrics.Average("mse"), mae=nnx.metrics.Average("mae"))
    
    writer = None
    if args.tensorboard:
        writer = SummaryWriter(path_to_target / "runs")
    
    for epoch in range(args.epochs):
        for x_batch, y_batch in train_loader:
            x, y = prepare_data(x_batch, y_batch, graph_data)
            train_step(model, optimiser, train_metric, x, y, decay=args.time_decay, alpha=args.alpha)
        for x_batch, y_batch in valid_loader:
            x, y = prepare_data(x_batch, y_batch, graph_data)
            evaluate(model, valid_metric, x, y)
        log_results_train(train_metric, valid_metric, epoch + 1, writer)
        train_metric.reset()
        valid_metric.reset()
    show_results_test(path_to_target / "plots", model, test_loader, graph_data, *args.nodes)
    
    if args.tensorboard:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
