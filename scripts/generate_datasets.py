import argparse
import re
import sys
from collections.abc import Collection
from functools import partial
from pathlib import Path
from typing import Any

sys.path.append(Path("src").absolute().as_posix())

import jax.numpy as jnp
import jax.tree_util as jtree
import numpy as np
import pandas as pd
from tqdm import tqdm

from traffic_flow_prediction.adjacency import (
    ADJType,
    generate_closeness,
    generate_correlation,
)
from traffic_flow_prediction.clustering import (
    CLType,
    compute_centroids,
    generate_base,
    generate_behaviour,
    generate_proximity,
    intersect,
)
from traffic_flow_prediction.datasets import (
    create_windows,
    split_time,
    to_edges,
)

WEEKDAY = {
    0: "monday",
    1: "tuesday",
    2: "wednesday",
    3: "thursday",
    4: "friday",
    5: "saturday",
    6: "sunday",
}

TFS = np.asarray([f"{i:02d}-{i+2:02d}" for i in range(0, 24, 2)])

fprint = partial(print, flush=True)


def fprint_tree(tree: Any, name: str = "tree", shape: bool = True) -> None:
    if shape:
        jtree.tree_map_with_path(lambda p, v: fprint(f"{name}{jtree.keystr(p)}.shape = {v.shape}"), tree)
    else:
        jtree.tree_map_with_path(lambda p, v: fprint(f"{name}{jtree.keystr(p)}\n{v}"), tree)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",      type=str,   default="data")
    parser.add_argument("--target",      type=str,   default="data")
    parser.add_argument("--window",      type=int,   default=12)
    parser.add_argument("--horizon",     type=int,   default=1)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    return parser.parse_args()


def read_traffic_csv(path_to_file: Path) -> pd.DataFrame:
    return pd.read_csv(
        path_to_file,
        sep=";",
        usecols=["id", "fecha", "tipo_elem", "intensidad", "ocupacion"],
        dtype={
            "id":         np.int64,
            "tipo_elem":  "category",
            "intensidad": np.float32,
            "ocupacion":  np.float32,
        },
        parse_dates=["fecha"],
    ).rename(columns={
        "id":         "id",
        "fecha":      "date",
        "tipo_elem":  "type",
        "intensidad": "intensity",
        "ocupacion":  "occupation",
    })


def read_sensors_csv(path_to_file: Path) -> pd.DataFrame:
    return pd.read_csv(
        path_to_file,
        sep=";",
        encoding="latin-1",
        usecols=["id", "distrito", "nombre", "utm_x", "utm_y", "longitud", "latitud"],
        dtype={
            "id":       np.int64,
            "distrito": "str",
            "nombre":   "str",
            "utm_x":    np.float32,
            "utm_y":    np.float32,
            "longitud": np.float32,
            "latitud":  np.float32,
        }
    ).rename(columns={
        "id":       "id",
        "distrito": "district",
        "nombre":   "name",
        "utm_x":    "utm_x",
        "utm_y":    "utm_y",
        "longitud": "longitude",
        "latitud":  "latitude",
    })


def prepare_traffic(traffic: pd.DataFrame) -> pd.DataFrame:
    df = traffic.dropna(subset=["id", "date"])
    
    congestion = df["occupation"] / df["intensity"].replace(0, np.nan)
    upper = np.nanpercentile(congestion, 99.0)
    congestion = congestion.clip(upper=upper)
    df["congestion"] = congestion.fillna(0.0)
    
    dt = df["date"].dt
    df["year"]       = dt.year
    df["month"]      = dt.month
    df["day"]        = dt.day
    df["hour"]       = dt.hour
    df["minute"]     = dt.minute
    dayofweek = dt.dayofweek
    df["weekday"]    = dayofweek.map(WEEKDAY)
    df["daytype"]    = np.where(dayofweek < 5, "WD", "WE")
    df["TF"] = TFS[df["hour"] // 2]
    df["daytype_TF"] = df["daytype"] + " " + df["TF"]
    
    df["weekday"] = df["weekday"].astype("category")
    df["daytype"] = df["daytype"].astype("category")
    df["TF"] = df["TF"].astype("category")
    df["daytype_TF"] = df["daytype_TF"].astype("category")
    
    return df


def prepare_sensors(sensors: pd.DataFrame) -> pd.DataFrame:
    df = sensors.dropna(subset=["id", "utm_x", "utm_y"])
    return df


def load_data(path_to_source: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path_to_dir_traffic = path_to_source / "raw" / "traffic"
    dfs = []
    files = sorted(path_to_dir_traffic.rglob("*.csv"))
    for path_to_file in tqdm(files, desc="Loading traffic data"):
        df = read_traffic_csv(path_to_file)
        dfs.append(prepare_traffic(df))
    traffic = pd.concat(dfs, ignore_index=True)
    
    path_to_dir_sensors = path_to_source / "raw" / "sensors"
    dfs = []
    files = sorted(path_to_dir_sensors.rglob("*.csv"))
    for path_to_file in tqdm(files, desc="Loading sensors data"):
        df = read_sensors_csv(path_to_file)
        dfs.append(prepare_sensors(df))
    sensors = pd.concat(dfs, ignore_index=True)
    
    return traffic, sensors


def build_clusters(traffic: pd.DataFrame, sensors: pd.DataFrame, cluster_targets: Collection[int] = (50, 500)) -> dict[str, pd.DataFrame]:
    clusters: dict[str, pd.DataFrame] = {}
    
    clusters[CLType.base.name] = generate_base(sensors)
    
    cluster_behaviour = generate_behaviour(traffic, 2)
    for target in cluster_targets:
        cluster_proximity = generate_proximity(sensors, target)
        clusters[f"{CLType.proximity.name}_{target}"] = cluster_proximity
        clusters[f"{CLType.proximity_behaviour.name}_{target}"] = intersect(cluster_proximity, cluster_behaviour)
    
    return clusters


def get_cluster_time_matrix(
    cluster: pd.DataFrame,
    cl_type: CLType, 
    traffic: pd.DataFrame,
) -> pd.DataFrame:
    traffic_cluster = traffic.merge(cluster[["id", cl_type.value, "utm_x", "utm_y"]], on="id", how="inner")
    cluster_time_matrix = (
        traffic_cluster
        .groupby([cl_type.value, "date"])["congestion"]
        .mean()
        .reset_index()
        .rename(columns={cl_type.value: "cluster"})
        .pivot(index="date", columns="cluster", values="congestion")
        .sort_index()
        .interpolate(limit_direction="both")
    )
    return cluster_time_matrix


def build_and_save_dataset(
    path_to_target: Path,
    cluster:        pd.DataFrame,
    cl_name:        str,
    traffic:        pd.DataFrame,
    *,
    window:      int,
    horizon:     int,
    train_ratio: float,
) -> None:
    cl_type = CLType[re.sub(r"_\d+$", "", cl_name)]
    c_t_matrix = get_cluster_time_matrix(cluster, cl_type, traffic)
    for adj_type in ADJType:
        fprint(f"Building {cl_name}/{adj_type.name}...", end=" ")
        match adj_type:
            case ADJType.correlation:
                adj_fct = generate_correlation
            case ADJType.closeness:
                adj_fct = partial(generate_closeness, centroids=compute_centroids(cluster, cl_type))
        matrix_gnn, df_edges, nodes_idx = adj_fct(c_t_matrix, 5)
        nodes_idx = jnp.asarray(nodes_idx, dtype=jnp.int32)
        edges, receivers, senders = to_edges(df_edges)
        data = jnp.asarray(matrix_gnn.values, dtype=jnp.float32)
        x, y = create_windows(data, window, horizon)
        # x_train, y_train, x_valid, y_valid, x_test, y_test = split_time(x, y, train_ratio)
        x_train, x_valid, x_test = split_time(x, train_ratio)
        y_train, y_valid, y_test = split_time(y, train_ratio)
        
        path_to_dir = path_to_target / cl_name / adj_type.name
        path_to_dir.mkdir(parents=True, exist_ok=True)
        jnp.savez(path_to_dir / "train.npz", x=x_train, y=y_train)
        jnp.savez(path_to_dir / "valid.npz", x=x_valid, y=y_valid)
        jnp.savez(path_to_dir / "test.npz",  x=x_test,  y=y_test)
        jnp.savez(path_to_dir / "graph.npz", edges=edges, receivers=receivers, senders=senders, nodes_idx=nodes_idx)
        matrix_gnn.to_csv(path_to_dir / "matrix_gnn.csv")
        df_edges.to_csv(path_to_dir / "df_edges.csv", index=False)
        fprint("OK")


def main() -> None:
    args = get_args()
    path_to_source = Path(args.source)
    path_to_target = Path(args.target)
    traffic, sensors = load_data(path_to_source)
    fprint("Building different clusterings...", end=" ")
    clusters = build_clusters(traffic, sensors)
    fprint("OK")
    for cl_name, cluster in clusters.items():
        build_and_save_dataset(
            path_to_target / "processed",
            cluster,
            cl_name,
            traffic,
            window=args.window,
            horizon=args.horizon,
            train_ratio=args.train_ratio,
        )


if __name__ == "__main__":
    main()