from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class ClusterSamplingConfig:
    run_dir: str = "C:/Users/spaeth/Desktop/Projecte/Konferenzen/ASAB/ASAB/CVAutoEncoder/runs/cvae_mnist"
    latent_file: str = "train_latent_mu.npy"
    labels_file: str = "train_labels.npy"

    # label-based clusters
    num_classes: int = 10

    # sampling
    epsilon: float = 1.0
    n_samples_per_cluster: int = 500
    random_state: int = 42

    # radius
    use_quantile_radius: bool = False
    radius_quantile: float = 0.95

    # outputs
    output_subdir: str = "cluster_sampling_by_label"


def validate_epsilon(epsilon: float) -> None:
    if not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be a float.")
    if not (0.0 <= float(epsilon) <= 1.0):
        raise ValueError("epsilon must be in [0, 1].")


def validate_n_samples(n_samples: int) -> None:
    if not isinstance(n_samples, int):
        raise TypeError("n_samples must be an integer.")
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0.")


def load_latent_vectors(run_dir: str | Path, latent_file: str) -> np.ndarray:
    path = Path(run_dir) / latent_file
    if not path.exists():
        raise FileNotFoundError(f"Missing latent file: {path}")

    X = np.load(path).astype(np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected latent vectors shape (N, D), got {X.shape}")

    print(f"Loaded latent vectors from: {path}")
    print(f"Latent shape: {X.shape}")
    return X


def load_labels(run_dir: str | Path, labels_file: str) -> np.ndarray:
    path = Path(run_dir) / labels_file
    if not path.exists():
        raise FileNotFoundError(f"Missing labels file: {path}")

    y = np.load(path).astype(np.int32).reshape(-1)
    print(f"Loaded labels from: {path}")
    print(f"Labels shape: {y.shape}")
    return y


def group_vectors_by_label(
    latent_vectors: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> Dict[int, np.ndarray]:
    if len(latent_vectors) != len(labels):
        raise ValueError(
            f"latent_vectors and labels must have same length, got "
            f"{len(latent_vectors)} and {len(labels)}"
        )

    grouped: Dict[int, np.ndarray] = {}
    for digit in range(num_classes):
        grouped[digit] = latent_vectors[labels == digit]
    return grouped


def compute_centroids(grouped_vectors: Dict[int, np.ndarray], latent_dim: int) -> np.ndarray:
    centroids = []
    for digit in sorted(grouped_vectors.keys()):
        vectors = grouped_vectors[digit]
        if len(vectors) == 0:
            centroid = np.zeros((latent_dim,), dtype=np.float32)
        else:
            centroid = np.mean(vectors, axis=0).astype(np.float32)
        centroids.append(centroid)
    return np.stack(centroids, axis=0)


def compute_cluster_radius(
    vectors: np.ndarray,
    centroid: np.ndarray,
    use_quantile: bool = False,
    quantile: float = 0.95,
) -> float:
    if len(vectors) == 0:
        return 0.0

    distances = np.linalg.norm(vectors - centroid, axis=1)

    if use_quantile:
        return float(np.quantile(distances, quantile))
    return float(np.max(distances))


def compute_cluster_radii(
    grouped_vectors: Dict[int, np.ndarray],
    centroids: np.ndarray,
    use_quantile: bool = False,
    quantile: float = 0.95,
) -> np.ndarray:
    radii = []
    for digit in range(len(centroids)):
        radius = compute_cluster_radius(
            vectors=grouped_vectors[digit],
            centroid=centroids[digit],
            use_quantile=use_quantile,
            quantile=quantile,
        )
        radii.append(radius)
    return np.array(radii, dtype=np.float32)


def sample_from_epsilon_ball_old(
    centroid: np.ndarray,
    radius: float,
    radius_old : float,
    n_samples: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    validate_n_samples(n_samples)

    centroid = np.asarray(centroid, dtype=np.float32)
    if centroid.ndim != 1:
        raise ValueError("centroid must be 1D.")
    if radius < 0:
        raise ValueError("radius must be >= 0.")

    dim = centroid.shape[0]
    rng = np.random.default_rng(random_state)

    if radius == 0:
        return np.tile(centroid, (n_samples, 1))


    directions = rng.normal(size=(n_samples, dim)).astype(np.float32)
    norms = np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
    directions = directions / norms


    # inner / outer radius
    r_inner = radius_old
    r_outer = radius


    # uniform sampling in shell
    u = rng.random(n_samples).astype(np.float32)

    scales = (
        u * (r_outer**dim - r_inner**dim) + r_inner**dim
    ) ** (1.0 / dim)


    samples = centroid + directions * scales[:, None]

    return samples.astype(np.float32)

def sample_from_epsilon_ball(
    centroid: np.ndarray,
    radius: float,
    radius_old: float,
    n_samples: int,
    random_state: int | None = None,
) -> np.ndarray:

    validate_n_samples(n_samples)

    centroid = np.asarray(centroid, dtype=np.float32)
    dim = centroid.shape[0]
    rng = np.random.default_rng(random_state)

    if radius == 0:
        return np.tile(centroid, (n_samples, 1)).astype(np.float32)

    # 1) polar coordinates
    directions = rng.normal(size=(n_samples, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    r_inner = max(0.0, radius_old)
    r_outer = radius
    print(r_inner, r_outer)
    radius = rng.uniform(r_inner,r_outer,n_samples).astype(np.float32)

    # 3) recreate points
    samples = centroid + directions * radius[:, None]
    dist = np.linalg.norm(samples - centroid, axis=1)

    print("expected:", r_inner, r_outer)
    print("actual:  ", dist.min(), dist.max())


    return samples.astype(np.float32)

def monte_carlo_sample_cluster(
    centroid: np.ndarray,
    cluster_radius: float,
    epsilon: float,
    epsilon_step: float,
    n_samples: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    validate_epsilon(epsilon)
    validate_n_samples(n_samples)

    sampling_radius = epsilon * cluster_radius
    sampling_radius_old = (epsilon - epsilon_step) * cluster_radius
    return sample_from_epsilon_ball(
        centroid=centroid,
        radius=sampling_radius,
        radius_old=sampling_radius_old,
        n_samples=n_samples,
        random_state=random_state,
    )


def monte_carlo_sample_all_clusters(
    centroids: np.ndarray,
    radii: np.ndarray,
    epsilon: float,
    epsilon_step: float,
    n_samples_per_cluster: int,
    random_state: int = 42,
) -> Dict[int, np.ndarray]:
    validate_epsilon(epsilon)
    validate_n_samples(n_samples_per_cluster)

    rng = np.random.default_rng(random_state)
    sampled = {}

    for digit in range(len(centroids)):
        seed = int(rng.integers(0, 1_000_000_000))
        sampled[digit] = monte_carlo_sample_cluster(
            centroid=centroids[digit],
            cluster_radius=float(radii[digit]),
            epsilon=epsilon,
            epsilon_step=epsilon_step,
            n_samples=n_samples_per_cluster,
            random_state=seed,
        )

    return sampled


def save_cluster_outputs(
    output_dir: str | Path,
    centroids: np.ndarray,
    radii: np.ndarray,
    sampled_points: Dict[int, np.ndarray],
) -> None:
    output_dir = Path(output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "cluster_centroids.npy", centroids)
    np.save(output_dir / "cluster_radii.npy", radii)

    for digit, points in sampled_points.items():
        np.save(output_dir / f"cluster_{digit}_samples.npy", points)
        np.save(output_dir / f"cluster_{digit}_label.npy", np.array([digit], dtype=np.int32))

    print(f"Saved centroids to: {output_dir / 'cluster_centroids.npy'}")
    print(f"Saved radii to: {output_dir / 'cluster_radii.npy'}")
    print(f"Saved sampled latent points in: {output_dir}")


def print_cluster_summary(
    grouped_vectors: Dict[int, np.ndarray],
    radii: np.ndarray,
    centroids: np.ndarray,
) -> None:
    print("\nLabel-based cluster summary:")
    for digit in range(len(centroids)):
        print(
            f"  digit {digit}: "
            f"n_vectors={len(grouped_vectors[digit])}, "
            f"radius={radii[digit]:.6f}, "
            f"centroid_shape={centroids[digit].shape}"
        )


def main():
    # -------------------------------------------------
    # Epsilon Schedule
    # -------------------------------------------------
    epsilon_step = 0.1
    epsilon_start = 0.0
    epsilon_end = 1.0
    epsilons = np.arange(epsilon_start, epsilon_end + epsilon_step, epsilon_step)


    config = ClusterSamplingConfig(
        run_dir="C:/Users/spaeth/Desktop/Projecte/Konferenzen/ASAB/ASAB/CVAutoEncoder/runs/cvae_mnist",
        latent_file="train_latent_mu.npy",
        labels_file="train_labels.npy",
        num_classes=10,
        epsilon=0.0,   # changed later on
        n_samples_per_cluster=1000,
        random_state=42,
        use_quantile_radius=False,
        radius_quantile=0.95,
        output_subdir="cluster_sampling_by_label_new_area",
    )

    # -------------------------------------------------
    # Load once (effizient)
    # -------------------------------------------------
    X = load_latent_vectors(config.run_dir, config.latent_file)
    y = load_labels(config.run_dir, config.labels_file)

    grouped_vectors = group_vectors_by_label(
        latent_vectors=X,
        labels=y,
        num_classes=config.num_classes,
    )

    centroids = compute_centroids(
        grouped_vectors=grouped_vectors,
        latent_dim=X.shape[1],
    )

    radii = compute_cluster_radii(
        grouped_vectors=grouped_vectors,
        centroids=centroids,
        use_quantile=config.use_quantile_radius,
        quantile=config.radius_quantile,
    )

    print_cluster_summary(
        grouped_vectors=grouped_vectors,
        radii=radii,
        centroids=centroids,
    )

    # -------------------------------------------------
    # Output Dir
    # -------------------------------------------------
    output_dir = Path(config.run_dir) / config.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)


    # -------------------------------------------------
    # Loop
    # -------------------------------------------------
    for epsilon in epsilons:

        print(f"\n===== Sampling with epsilon={epsilon:.2f} =====")

        sampled_points = monte_carlo_sample_all_clusters(
            centroids=centroids,
            radii=radii,
            epsilon=epsilon,
            epsilon_step=epsilon_step,
            n_samples_per_cluster=config.n_samples_per_cluster,
            random_state=config.random_state,
        )

        # -------------------------------
        # Save CSV per cluster
        # -------------------------------
        epsilon_dir = output_dir / f"epsilon_{epsilon:.2f}"
        epsilon_dir.mkdir(parents=True, exist_ok=True)

        for digit, points in sampled_points.items():

            print(f"digit {digit} sampled shape: {points.shape}")

            df = pd.DataFrame(points)

            file_path = epsilon_dir / f"samples_digit_{digit}_epsilon_{epsilon:.2f}.csv"
            df.to_csv(file_path, index=False)


    print("\nDone.")
    print(f"Number of label-clusters: {config.num_classes}")
    print(f"Epsilon range: {epsilon_start} -> {epsilon_end}")
    print(f"Step size: {epsilon_step}")
    print(f"Samples per cluster: {config.n_samples_per_cluster}")

if __name__ == "__main__":
    main()