from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


@dataclass
class LoadConfig:
    run_dir: str = "./runs/cvae_mnist"
    prefix: str = "train"   # "train" or "valid"
    show_first_n: int = 5
    use_pca_if_needed: bool = True
    save_plot: bool = True
    plot_name: Optional[str] = None


def load_latent_files(run_dir: str | Path, prefix: str = "train"):
    run_dir = Path(run_dir)

    mu_path = run_dir / f"{prefix}_latent_mu.npy"
    log_var_path = run_dir / f"{prefix}_latent_log_var.npy"
    z_path = run_dir / f"{prefix}_latent_z.npy"
    labels_path = run_dir / f"{prefix}_labels.npy"

    if not mu_path.exists():
        raise FileNotFoundError(f"Missing file: {mu_path}")
    if not log_var_path.exists():
        raise FileNotFoundError(f"Missing file: {log_var_path}")
    if not z_path.exists():
        raise FileNotFoundError(f"Missing file: {z_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing file: {labels_path}")

    mu = np.load(mu_path)
    log_var = np.load(log_var_path)
    z = np.load(z_path)
    labels = np.load(labels_path)

    return mu, log_var, z, labels


def print_latent_summary(mu: np.ndarray, log_var: np.ndarray, z: np.ndarray, labels: np.ndarray, show_first_n: int = 5):
    print("Loaded latent files successfully.\n")

    print("Shapes:")
    print("  mu      :", mu.shape)
    print("  log_var :", log_var.shape)
    print("  z       :", z.shape)
    print("  labels  :", labels.shape)

    print("\nFirst few labels:")
    print(labels[:show_first_n])

    print("\nFirst few mu vectors:")
    for i in range(min(show_first_n, len(mu))):
        print(f"mu[{i}] = {mu[i]}")

    print("\nLatent statistics:")
    print(f"  mu mean: {mu.mean():.6f}")
    print(f"  mu std : {mu.std():.6f}")
    print(f"  z mean : {z.mean():.6f}")
    print(f"  z std  : {z.std():.6f}")


def plot_latent_space_2d(
    latent: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: Optional[str | Path] = None,
):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent[:, 0], latent[:, 1], c=labels, s=8)
    plt.colorbar(scatter, label="Digit label")
    plt.title(title)
    plt.xlabel("Latent dimension 1")
    plt.ylabel("Latent dimension 2")
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()
    plt.close()


def visualize_latent_vectors(
    mu: np.ndarray,
    labels: np.ndarray,
    run_dir: str | Path,
    prefix: str,
    use_pca_if_needed: bool = True,
    save_plot: bool = True,
    plot_name: Optional[str] = None,
):
    run_dir = Path(run_dir)
    latent_dim = mu.shape[1]

    if latent_dim == 2:
        save_path = None
        if save_plot:
            save_path = run_dir / (plot_name or f"{prefix}_latent_mu_2d.png")

        plot_latent_space_2d(
            latent=mu,
            labels=labels,
            title=f"{prefix} latent space (mu)",
            save_path=save_path,
        )
        return

    if not use_pca_if_needed:
        print(f"Latent dimension is {latent_dim}, so direct 2D plotting is not possible without PCA.")
        return

    print(f"\nApplying PCA: {latent_dim}D -> 2D")
    pca = PCA(n_components=2, random_state=42)
    mu_2d = pca.fit_transform(mu)

    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    save_path = None
    if save_plot:
        save_path = run_dir / (plot_name or f"{prefix}_latent_mu_pca_2d.png")

    plot_latent_space_2d(
        latent=mu_2d,
        labels=labels,
        title=f"{prefix} latent space (mu) projected with PCA",
        save_path=save_path,
    )


def main():
    config = LoadConfig(
        run_dir="./runs/cvae_mnist",
        prefix="train",
        show_first_n=5,
        use_pca_if_needed=True,
        save_plot=True,
        plot_name=None,
    )

    mu, log_var, z, labels = load_latent_files(
        run_dir=config.run_dir,
        prefix=config.prefix,
    )

    print_latent_summary(
        mu=mu,
        log_var=log_var,
        z=z,
        labels=labels,
        show_first_n=config.show_first_n,
    )

    visualize_latent_vectors(
        mu=mu,
        labels=labels,
        run_dir=config.run_dir,
        prefix=config.prefix,
        use_pca_if_needed=config.use_pca_if_needed,
        save_plot=config.save_plot,
        plot_name=config.plot_name,
    )


if __name__ == "__main__":
    main()