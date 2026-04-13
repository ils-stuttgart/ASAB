from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass
class Cluster:
    """
    Represents one cluster with its member vectors and centroid.

    Attributes
    ----------
    vectors : np.ndarray
        Array of shape (n_vectors, n_features).
    centroid : np.ndarray
        Array of shape (n_features,).
    """
    vectors: np.ndarray
    centroid: np.ndarray

    def __post_init__(self) -> None:
        self.vectors = np.asarray(self.vectors, dtype=float)
        self.centroid = np.asarray(self.centroid, dtype=float)

        if self.vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array of shape (n_vectors, n_features).")
        if self.centroid.ndim != 1:
            raise ValueError("centroid must be a 1D array of shape (n_features,).")
        if self.vectors.shape[1] != self.centroid.shape[0]:
            raise ValueError(
                "Dimension mismatch: vectors.shape[1] must equal centroid.shape[0]."
            )


def _validate_epsilon(epsilon: float) -> None:
    if not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be a float in [0, 1].")
    if not (0.0 <= float(epsilon) <= 1.0):
        raise ValueError("epsilon must be in the range [0, 1].")


def _validate_n_samples(n_samples: int) -> None:
    if not isinstance(n_samples, int):
        raise TypeError("n_samples must be an integer.")
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0.")


def _compute_cluster_radius(vectors: np.ndarray, centroid: np.ndarray) -> float:
    """
    Computes the maximum Euclidean distance from the centroid to cluster vectors.
    """
    distances = np.linalg.norm(vectors - centroid, axis=1)
    return float(np.max(distances)) if len(distances) > 0 else 0.0


def sample_from_epsilon_ball(
    centroid: np.ndarray,
    radius: float,
    n_samples: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Uniformly sample points inside a d-dimensional ball.

    Parameters
    ----------
    centroid : np.ndarray
        Center of the ball, shape (n_features,).
    radius : float
        Radius of the ball.
    n_samples : int
        Number of samples.
    random_state : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Sampled points of shape (n_samples, n_features).
    """
    centroid = np.asarray(centroid, dtype=float)
    if centroid.ndim != 1:
        raise ValueError("centroid must be a 1D array.")
    _validate_n_samples(n_samples)

    if radius < 0:
        raise ValueError("radius must be >= 0.")

    rng = np.random.default_rng(random_state)
    dim = centroid.shape[0]

    if radius == 0:
        return np.tile(centroid, (n_samples, 1))

    directions = rng.normal(size=(n_samples, dim))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / norms

    scales = rng.random(n_samples) ** (1.0 / dim)
    samples = centroid + directions * (scales[:, None] * radius)

    return samples


def monte_carlo_sample_cluster(
    cluster: Cluster,
    epsilon: float,
    n_samples: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Monte Carlo sampling around a cluster centroid within an epsilon-ball.

    The effective sampling radius is:
        sampling_radius = epsilon * cluster_radius

    where cluster_radius is the maximum distance from the centroid to any vector
    in the cluster.

    Parameters
    ----------
    cluster : Cluster
        Cluster object containing vectors and centroid.
    epsilon : float
        Scaling factor in [0, 1].
        - 0.0: sample exactly at centroid
        - 1.0: sample within the full cluster radius
    n_samples : int
        Number of Monte Carlo samples.
    random_state : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Sampled vectors of shape (n_samples, n_features).
    """
    _validate_epsilon(epsilon)
    _validate_n_samples(n_samples)

    cluster_radius = _compute_cluster_radius(cluster.vectors, cluster.centroid)
    sampling_radius = epsilon * cluster_radius

    return sample_from_epsilon_ball(
        centroid=cluster.centroid,
        radius=sampling_radius,
        n_samples=n_samples,
        random_state=random_state,
    )


def monte_carlo_sample_multiple_clusters(
    clusters: Iterable[Cluster],
    epsilon: float,
    n_samples_per_cluster: int,
    random_state: Optional[int] = None,
) -> dict[int, np.ndarray]:
    """
    Monte Carlo sample around multiple cluster centroids.

    Parameters
    ----------
    clusters : Iterable[Cluster]
        Collection of Cluster objects.
    epsilon : float
        Scaling factor in [0, 1].
    n_samples_per_cluster : int
        Number of samples to draw for each cluster.
    random_state : Optional[int]
        Random seed.

    Returns
    -------
    dict[int, np.ndarray]
        Dictionary mapping cluster index to sampled points.
    """
    _validate_epsilon(epsilon)
    _validate_n_samples(n_samples_per_cluster)

    rng = np.random.default_rng(random_state)
    results: dict[int, np.ndarray] = {}

    for idx, cluster in enumerate(clusters):
        seed = int(rng.integers(0, 1_000_000_000))
        results[idx] = monte_carlo_sample_cluster(
            cluster=cluster,
            epsilon=epsilon,
            n_samples=n_samples_per_cluster,
            random_state=seed,
        )

    return results