from enum import Enum

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class ADJType(Enum):
    correlation = "adj_correlation"
    closeness   = "adj_closeness"


def reindex_cluster_time_matrix(c_t_matrix: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int], list[int]]:
    matrix = c_t_matrix.copy()
    nodes_idx = sorted(matrix.columns.tolist())
    mapping = {cluster: i for i, cluster in enumerate(nodes_idx)}
    matrix = matrix[nodes_idx].copy()
    matrix.columns = [mapping[c] for c in matrix.columns]
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)
    return matrix, mapping, nodes_idx


def generate_correlation(
    c_t_matrix: pd.DataFrame,
    k:          int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    matrix = c_t_matrix.copy()
    
    std_per_cluster = matrix.std()
    constant_clusters = std_per_cluster[std_per_cluster == 0].index.tolist()
    matrix = matrix.drop(columns=constant_clusters)
    
    matrix_gnn, _, nodes_idx = reindex_cluster_time_matrix(matrix)
    corr_matrix = c_t_matrix.corr(method="pearson").fillna(0.0)
    corr_values = corr_matrix.values
    cluster_labels = corr_matrix.columns.tolist()
    
    edges = []
    n = corr_values.shape[0]
    if n == 1:
        edges.append((0, 0, 1.0))
    else:
        k_eff = min(k, n - 1)
        for i in range(n):
            queue = corr_values[i].copy()
            queue[i] = -np.inf
            top_idx = np.argsort(queue)[-k_eff:]
            top_idx = top_idx[np.argsort(queue[top_idx])[::-1]]
            source_cluster = cluster_labels[i]
            for j in top_idx:
                target_cluster = cluster_labels[j]
                weight = float(max(corr_values[i, j], 0.0))
                edges.append((source_cluster, target_cluster, weight))
    df_edges = pd.DataFrame(edges, columns=["source", "target", "weight"])
    return matrix_gnn, df_edges, nodes_idx


def knn_from_centroids(centroids: pd.DataFrame, k: int) -> pd.DataFrame:
    centroids = centroids.sort_values("cluster").reset_index(drop=True).copy()
    coords = centroids[["utm_x", "utm_y"]]
    n = len(coords)
    if n == 0:
        raise ValueError("There is no centroids to build the adjacency")
    if n == 1:
        return pd.DataFrame({"source": [0], "target": [0], "distance": [0.0], "weight": [1.0]})
    
    k_eff = min(k, n - 1)
    model = NearestNeighbors(n_neighbors=k_eff+1, metric="euclidean")
    model.fit(coords)
    distances, indices = model.kneighbors(coords)
    
    edges = []
    for i in range(n):
        source = int(centroids.loc[i, "cluster"])
        for dist, j in zip(distances[i, 1:], indices[i, 1:]):
            target = int(centroids.loc[j, "cluster"])
            edges.append((source, target, float(dist)))
    
    df_edges = pd.DataFrame(edges, columns=["source", "target", "distance"])
    sigma = df_edges["distance"].replace(0, np.nan).mean()
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    df_edges["weight"] = np.exp(-df_edges["distance"] / sigma)
    return df_edges


def generate_closeness(
    c_t_matrix: pd.DataFrame,
    k:          int,
    centroids:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    matrix = c_t_matrix.copy()
    matrix_gnn, mapping, nodes_idx = reindex_cluster_time_matrix(matrix)
    
    centroids = centroids[centroids["cluster"].isin(mapping.keys())].copy()
    centroids["cluster"] = centroids["cluster"].map(mapping).astype(int)
    centroids = centroids.sort_values("cluster")
    
    df_edges = knn_from_centroids(centroids, k)
    
    return matrix_gnn, df_edges, nodes_idx
