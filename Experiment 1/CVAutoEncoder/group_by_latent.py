from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np


@dataclass
class GroupConfig:
    run_dir: str = "./runs/cvae_mnist"
    prefix: str = "train"   # "train" or "valid"
    num_classes: int = 10
    save_centroids: bool = True
    save_indices: bool = True


def load_latent_and_labels(run_dir: str | Path, prefix: str = "train"):
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


def group_latents_by_label(
    mu: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 10,
) -> Dict[int, np.ndarray]:
    grouped = {}

    for digit in range(num_classes):
        grouped[digit] = mu[labels == digit]

    return grouped


def compute_class_centroids(grouped: Dict[int, np.ndarray], latent_dim: int) -> np.ndarray:
    centroids = []

    for digit in sorted(grouped.keys()):
        vectors = grouped[digit]
        if len(vectors) == 0:
            centroid = np.zeros((latent_dim,), dtype=np.float32)
        else:
            centroid = np.mean(vectors, axis=0).astype(np.float32)
        centroids.append(centroid)

    return np.stack(centroids, axis=0)


def save_grouped_latents(
    grouped: Dict[int, np.ndarray],
    mu: np.ndarray,
    labels: np.ndarray,
    run_dir: str | Path,
    prefix: str = "train",
    save_indices: bool = True,
) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    for digit, vectors in grouped.items():
        np.save(run_dir / f"{prefix}_digit_{digit}_latent_mu.npy", vectors)

        if save_indices:
            indices = np.where(labels == digit)[0]
            np.save(run_dir / f"{prefix}_digit_{digit}_indices.npy", indices)

        print(f"Digit {digit}: saved {len(vectors)} latent vectors")


def save_centroids(
    centroids: np.ndarray,
    run_dir: str | Path,
    prefix: str = "train",
) -> None:
    run_dir = Path(run_dir)
    np.save(run_dir / f"{prefix}_digit_centroids.npy", centroids)
    print(f"Saved digit centroids: {run_dir / f'{prefix}_digit_centroids.npy'}")
    print(f"Centroids shape: {centroids.shape}")


def print_summary(grouped: Dict[int, np.ndarray]) -> None:
    print("\nGrouped latent vectors summary:")
    for digit, vectors in grouped.items():
        print(f"  digit {digit}: {vectors.shape}")


def main():
    config = GroupConfig(
        run_dir="./runs/cvae_mnist",
        prefix="train",
        num_classes=10,
        save_centroids=True,
        save_indices=True,
    )

    mu, labels = load_latent_and_labels(
        run_dir=config.run_dir,
        prefix=config.prefix,
    )

    print(f"Loaded mu shape     : {mu.shape}")
    print(f"Loaded labels shape : {labels.shape}")

    grouped = group_latents_by_label(
        mu=mu,
        labels=labels,
        num_classes=config.num_classes,
    )

    print_summary(grouped)

    save_grouped_latents(
        grouped=grouped,
        mu=mu,
        labels=labels,
        run_dir=config.run_dir,
        prefix=config.prefix,
        save_indices=config.save_indices,
    )

    if config.save_centroids:
        centroids = compute_class_centroids(
            grouped=grouped,
            latent_dim=mu.shape[1],
        )
        save_centroids(
            centroids=centroids,
            run_dir=config.run_dir,
            prefix=config.prefix,
        )


if __name__ == "__main__":
    main()