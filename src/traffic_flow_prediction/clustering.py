from enum import Enum

import pandas as pd
from sklearn.cluster import DBSCAN, KMeans


class CLType(Enum):
    base = "cl_base"
    proximity = "cl_proximity"
    behaviour = "cl_behaviour"
    proximity_behaviour = "cl_intersection"


def generate_base(sensors: pd.DataFrame) -> pd.DataFrame:
    df = sensors.copy()
    df[CLType.base.value] = pd.factorize(df["id"])[0]
    return df


def generate_behaviour(traffic: pd.DataFrame, n_clusters:  int) -> pd.DataFrame:
    df = (
        traffic
        .groupby(["id", "daytype_TF"])["congestion"]
        .mean()
        .reset_index()
        .pivot(index="id", columns="daytype_TF", values="congestion")
        .sort_index()
        .dropna()
    )
    
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(df)
    df[CLType.behaviour.value] = labels
    return df.reset_index()[["id", CLType.behaviour.value]]


def _iter_proximity(sensors: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    df = sensors.copy()
    coords = df[["utm_x", "utm_y"]]
    model = DBSCAN(eps=epsilon, min_samples=1, metric="euclidean")
    labels = pd.Series(model.fit_predict(coords), index=df.index)
    next_label = labels[labels >= 0].max() + 1 if (labels >= 0).any() else 0
    for idx in labels[labels == -1].index:
        labels.loc[idx] = next_label
        next_label += 1
    unique_labels = sorted(labels.unique())
    mapping = {old: new for new, old in enumerate(unique_labels)}
    df[CLType.proximity.value] = labels.map(mapping)
    return df


def generate_proximity(sensors: pd.DataFrame, n_clusters:  int, *,
                       n_iters: int = 25, eps_min: float = 10, eps_max: float = 1000) -> pd.DataFrame:
    best_df = None
    best_diff = float("inf")
    lb = eps_min
    ub = eps_max
    for _ in range(n_iters):
        mid = (lb + ub) / 2
        df = _iter_proximity(sensors, epsilon=mid)
        vals = df[CLType.proximity.value].dropna().unique()
        vals = [v for v in vals if int(v) != -1]
        nc = len(vals)
        diff = abs(n_clusters - nc)
        if diff < best_diff:
            best_df = df
            best_diff = diff
        if nc > n_clusters:
            lb = mid
        else:
            ub = mid
    return best_df


def intersect(
    cluster_proximity: pd.DataFrame,
    cluster_behaviour: pd.DataFrame,
) -> pd.DataFrame:
    df = cluster_proximity.merge(cluster_behaviour, on="id", how="inner")
    pairs = list(zip(df[CLType.proximity.value], df[CLType.behaviour.value]))
    df[CLType.proximity_behaviour.value] = pd.factorize(pd.Series(pairs, index=df.index))[0]
    return df


def reindex_cluster(
    cluster: pd.DataFrame,
    cl_type: CLType,
) -> tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
    df = cluster.copy()
    nodes = sorted(df[cl_type.value].dropna().unique().tolist())
    mapping     = {old: new for new, old in enumerate(nodes)}
    mapping_inv = {new: old for old, new in mapping.items()}
    df[cl_type.value] = df[cl_type.value].map(mapping).astype(int)
    return df, mapping, mapping_inv


def compute_centroids(cluster: pd.DataFrame, cl_type: CLType) -> pd.DataFrame:
    centroids = (
        cluster
        .groupby(cl_type.value)[["utm_x", "utm_y"]]
        .mean()
        .reset_index()
        .rename(columns={cl_type.value: "cluster"})
        .sort_values("cluster")
        .reset_index()
    )
    return centroids