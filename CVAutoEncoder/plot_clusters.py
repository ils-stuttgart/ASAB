from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


@dataclass
class PlotConfig:
    run_dir: str = "./runs/cvae_mnist"
    prefix: str = "train"
    num_classes: int = 10
    point_size: int = 8
    alpha: float = 0.7
    show_centroids: bool = True
    save_name: str = "digit_clusters_pca.png"


def load_latent_and_labels(run_dir: str | Path, prefix: str):
    run_dir = Path(run_dir)

    mu = np.load(run_dir / f"{prefix}_latent_mu.npy")
    labels = np.load(run_dir / f"{prefix}_labels.npy")

    return mu, labels


def compute_centroids_2d(points_2d: np.ndarray, labels: np.ndarray, num_classes: int):
    centroids = []

    for digit in range(num_classes):
        pts = points_2d[labels == digit]

        if len(pts) == 0:
            centroids.append(np.array([0.0, 0.0]))
        else:
            centroids.append(np.mean(pts, axis=0))

    return np.array(centroids)


def plot_clusters(
    points_2d: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    point_size: int,
    alpha: float,
    show_centroids: bool,
    save_path: str | Path,
):
    plt.figure(figsize=(10, 8))

    cmap = plt.cm.get_cmap("tab10", num_classes)

    for digit in range(num_classes):
        pts = points_2d[labels == digit]

        plt.scatter(
            pts[:, 0],
            pts[:, 1],
            s=point_size,
            alpha=alpha,
            color=cmap(digit),
            label=f"Digit {digit}",
        )

    if show_centroids:
        centroids = compute_centroids_2d(points_2d, labels, num_classes)

        plt.scatter(
            centroids[:, 0],
            centroids[:, 1],
            s=250,
            c="black",
            marker="X",
            label="Centroids",
        )

        for digit in range(num_classes):
            plt.text(
                centroids[digit, 0],
                centroids[digit, 1],
                str(digit),
                fontsize=12,
                weight="bold",
                color="white",
                ha="center",
                va="center",
            )

    plt.title("Latent Space Clusters by Digit (PCA Projection)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.legend(markerscale=2, fontsize=9)
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
    print("mu shape     :", mu.shape)
    print("labels shape :", labels.shape)

    # Reduce to 2D
    pca = PCA(n_components=2, random_state=42)
    points_2d = pca.fit_transform(mu)

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Total explained variance:", pca.explained_variance_ratio_.sum())

    save_path = Path(config.run_dir) / config.save_name

    plot_clusters(
        points_2d=points_2d,
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