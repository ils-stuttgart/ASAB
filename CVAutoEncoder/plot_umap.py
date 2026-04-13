from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# pip install umap-learn
import umap.umap_ as umap


@dataclass
class PlotConfig:
    run_dir: str = "./runs/cvae_mnist"
    prefix: str = "train"
    num_classes: int = 10

    # UMAP parameters
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"
    random_state: int = 42

    # Plot parameters
    point_size: int = 8
    alpha: float = 0.7
    show_centroids: bool = True
    save_name: str = "digit_clusters_umap_3d.png"


def load_latent_and_labels(run_dir: str | Path, prefix: str):
    run_dir = Path(run_dir)

    mu_path = run_dir / f"{prefix}_latent_mu.npy"
    labels_path = run_dir / f"{prefix}_labels.npy"

    if not mu_path.exists():
        raise FileNotFoundError(f"Missing file: {mu_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing file: {labels_path}")

    mu = np.load(mu_path)
    labels = np.load(labels_path)

    return mu, labels


def compute_centroids_3d(points_3d: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    centroids = []

    for digit in range(num_classes):
        pts = points_3d[labels == digit]
        if len(pts) == 0:
            centroids.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        else:
            centroids.append(np.mean(pts, axis=0).astype(np.float32))

    return np.stack(centroids, axis=0)


def fit_umap_3d(
    latent_vectors: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(latent_vectors)
    return embedding


def plot_clusters_3d(
    points_3d: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    point_size: int,
    alpha: float,
    show_centroids: bool,
    save_path: str | Path,
):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.cm.get_cmap("tab10", num_classes)

    for digit in range(num_classes):
        pts = points_3d[labels == digit]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=point_size,
            alpha=alpha,
            color=cmap(digit),
            label=f"Digit {digit}",
        )

    if show_centroids:
        centroids = compute_centroids_3d(points_3d, labels, num_classes)

        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            centroids[:, 2],
            s=50,
            c="black",
            marker="X",
            label="Centroids",
        )

        for digit in range(num_classes):
            ax.text(
                centroids[digit, 0],
                centroids[digit, 1],
                centroids[digit, 2],
                str(digit),
                color="black",
            )

    ax.set_title("Latent Space Clusters by Digit (3D UMAP)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")
    ax.legend(markerscale=2, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()


def main():
    config = PlotConfig()

    mu, labels = load_latent_and_labels(
        run_dir=config.run_dir,
        prefix=config.prefix,
    )

    print("Loaded:")
    print("  mu shape     :", mu.shape)
    print("  labels shape :", labels.shape)

    points_3d = fit_umap_3d(
        latent_vectors=mu,
        n_neighbors=config.n_neighbors,
        min_dist=config.min_dist,
        metric=config.metric,
        random_state=config.random_state,
    )

    print("3D embedding shape:", points_3d.shape)

    save_path = Path(config.run_dir) / config.save_name

    plot_clusters_3d(
        points_3d=points_3d,
        labels=labels,
        num_classes=config.num_classes,
        point_size=config.point_size,
        alpha=config.alpha,
        show_centroids=config.show_centroids,
        save_path=save_path,
    )

    print(f"Saved plot to: {save_path}")


if __name__ == "__main__":
    main()