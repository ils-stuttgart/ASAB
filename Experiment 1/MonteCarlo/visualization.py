from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from MonteCarlo.MonteCarlo import Cluster, _compute_cluster_radius
def visualize_cluster_and_samples(
    cluster: Cluster,
    samples: np.ndarray,
    epsilon: float,
    figsize: tuple = (8, 8),
    alpha_cluster: float = 0.7,
    alpha_samples: float = 0.5,
    show_circle: bool = True,
    title: str = "Cluster and Monte Carlo Samples",
) -> None:
    """
    Visualize a 2D cluster, its centroid, Monte Carlo samples,
    and the epsilon-ball boundary.

    Parameters
    ----------
    cluster : Cluster
        Cluster containing vectors and centroid.
    samples : np.ndarray
        Monte Carlo samples of shape (n_samples, 2).
    epsilon : float
        Epsilon in [0, 1], used to scale the cluster radius.
    figsize : tuple
        Figure size for matplotlib.
    alpha_cluster : float
        Transparency for cluster points.
    alpha_samples : float
        Transparency for sampled points.
    show_circle : bool
        Whether to draw the epsilon-ball boundary.
    title : str
        Plot title.
    """
    vectors = np.asarray(cluster.vectors, dtype=float)
    centroid = np.asarray(cluster.centroid, dtype=float)
    samples = np.asarray(samples, dtype=float)

    if vectors.shape[1] != 2:
        raise ValueError("Visualization only supports 2D vectors.")
    if centroid.shape[0] != 2:
        raise ValueError("Visualization only supports a 2D centroid.")
    if samples.shape[1] != 2:
        raise ValueError("Visualization only supports 2D samples.")

    cluster_radius = _compute_cluster_radius(vectors, centroid)
    sampling_radius = epsilon * cluster_radius

    fig, ax = plt.subplots(figsize=figsize)

    # Cluster points
    ax.scatter(
        vectors[:, 0],
        vectors[:, 1],
        label="Cluster vectors",
        alpha=alpha_cluster,
        s=50,
        marker="o",
    )

    # Sampled points
    ax.scatter(
        samples[:, 0],
        samples[:, 1],
        label="Monte Carlo samples",
        alpha=alpha_samples,
        s=30,
        marker="x",
    )

    # Centroid
    ax.scatter(
        centroid[0],
        centroid[1],
        label="Centroid",
        s=200,
        marker="*",
    )

    # Epsilon-ball
    if show_circle:
        circle = Circle(
            (centroid[0], centroid[1]),
            sampling_radius,
            fill=False,
            linewidth=2,
            linestyle="--",
        )
        ax.add_patch(circle)

    ax.set_title(title)
    ax.set_xlabel("Latent dimension 1")
    ax.set_ylabel("Latent dimension 2")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    plt.show()


def verify_samples_within_epsilon_ball(
    cluster: Cluster,
    samples: np.ndarray,
    epsilon: float,
    tolerance: float = 1e-10,
) -> bool:
    """
    Check whether all sampled points lie inside the epsilon-ball.

    Parameters
    ----------
    cluster : Cluster
        Cluster object.
    samples : np.ndarray
        Sampled points of shape (n_samples, n_features).
    epsilon : float
        Epsilon in [0, 1].
    tolerance : float
        Numerical tolerance.

    Returns
    -------
    bool
        True if all points are within the epsilon-ball, else False.
    """
    vectors = np.asarray(cluster.vectors, dtype=float)
    centroid = np.asarray(cluster.centroid, dtype=float)
    samples = np.asarray(samples, dtype=float)

    cluster_radius = _compute_cluster_radius(vectors, centroid)
    sampling_radius = epsilon * cluster_radius

    distances = np.linalg.norm(samples - centroid, axis=1)
    return np.all(distances <= sampling_radius + tolerance)


def get_sample_distances(cluster: Cluster, samples: np.ndarray) -> np.ndarray:
    """
    Return Euclidean distances from each sample to the centroid.
    """
    centroid = np.asarray(cluster.centroid, dtype=float)
    samples = np.asarray(samples, dtype=float)
    return np.linalg.norm(samples - centroid, axis=1)