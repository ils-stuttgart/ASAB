from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


@dataclass
class DecodeConfig:
    run_dir: str = "./runs/cvae_mnist"
    decoder_path: str = "./runs/cvae_mnist/decoder.keras"
    cluster_sampling_dir: str = "./runs/cvae_mnist/cluster_sampling_by_label"

    num_classes: int = 10

    output_dirname: str = "decoded_cluster_samples"

    save_grid: bool = True
    save_individual_images: bool = True
    save_numpy_arrays: bool = True

    max_images_in_grid: int = 64
    grid_ncols: int = 8


def load_decoder(decoder_path: str | Path) -> tf.keras.Model:
    decoder_path = Path(decoder_path)
    if not decoder_path.exists():
        raise FileNotFoundError(f"Decoder file not found: {decoder_path}")

    decoder = tf.keras.models.load_model(decoder_path, compile=False)
    print(f"Loaded decoder from: {decoder_path}")
    return decoder


def list_available_cluster_ids(cluster_sampling_dir: str | Path) -> List[int]:
    cluster_sampling_dir = Path(cluster_sampling_dir)
    if not cluster_sampling_dir.exists():
        raise FileNotFoundError(f"Cluster sampling directory not found: {cluster_sampling_dir}")

    cluster_ids = []
    for path in cluster_sampling_dir.glob("cluster_*_samples.npy"):
        name = path.stem  # e.g. cluster_3_samples
        parts = name.split("_")
        if len(parts) != 3:
            continue
        cluster_id = int(parts[1])
        cluster_ids.append(cluster_id)

    cluster_ids = sorted(cluster_ids)

    if not cluster_ids:
        raise FileNotFoundError(f"No cluster sample files found in: {cluster_sampling_dir}")

    print(f"Found cluster ids: {cluster_ids}")
    return cluster_ids


def load_cluster_samples(cluster_sampling_dir: str | Path, cluster_id: int) -> np.ndarray:
    path = Path(cluster_sampling_dir) / f"cluster_{cluster_id}_samples.npy"
    if not path.exists():
        raise FileNotFoundError(f"Cluster sample file not found: {path}")

    samples = np.load(path).astype(np.float32)
    if samples.ndim != 2:
        raise ValueError(
            f"Expected latent samples of shape (N, latent_dim), got {samples.shape} for cluster {cluster_id}"
        )

    print(f"Loaded cluster {cluster_id} latent samples: {samples.shape}")
    return samples


def load_cluster_label(cluster_sampling_dir: str | Path, cluster_id: int) -> int:
    path = Path(cluster_sampling_dir) / f"cluster_{cluster_id}_label.npy"
    if not path.exists():
        raise FileNotFoundError(f"Cluster label file not found: {path}")

    label = np.load(path).astype(np.int32).reshape(-1)
    if len(label) == 0:
        raise ValueError(f"Cluster label file is empty: {path}")

    return int(label[0])


def build_condition_vectors(label: int, n_samples: int, num_classes: int) -> np.ndarray:
    if label < 0 or label >= num_classes:
        raise ValueError(
            f"Label {label} is invalid for num_classes={num_classes}"
        )

    labels = np.full((n_samples,), label, dtype=np.int32)
    one_hot = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    return np.asarray(one_hot, dtype=np.float32)


def decode_latent_samples(
    decoder: tf.keras.Model,
    latent_samples: np.ndarray,
    class_label: int,
    num_classes: int,
) -> np.ndarray:
    cond = build_condition_vectors(class_label, len(latent_samples), num_classes)
    decoded = decoder.predict([latent_samples, cond], verbose=0)
    decoded = np.asarray(decoded, dtype=np.float32)
    decoded = np.clip(decoded, 0.0, 1.0)
    return decoded


def save_grid(
    images: np.ndarray,
    save_path: str | Path,
    max_images: int = 64,
    ncols: int = 8,
) -> None:
    save_path = Path(save_path)

    images = images[:max_images]
    n_images = len(images)
    if n_images == 0:
        return

    nrows = int(np.ceil(n_images / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for ax in axes.flat:
        ax.axis("off")

    for i in range(n_images):
        ax = axes.flat[i]
        img = images[i]

        if img.shape[-1] == 1:
            ax.imshow(img.squeeze(), cmap="gray")
        else:
            ax.imshow(img)

        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_individual_images(images: np.ndarray, cluster_dir: str | Path) -> None:
    cluster_dir = Path(cluster_dir)
    for i, img in enumerate(images):
        plt.figure(figsize=(2, 2))
        if img.shape[-1] == 1:
            plt.imshow(img.squeeze(), cmap="gray")
        else:
            plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(cluster_dir / f"sample_{i}.png", dpi=150)
        plt.close()


def save_cluster_outputs(
    base_output_dir: str | Path,
    cluster_id: int,
    decoded_images: np.ndarray,
    decode_label: int,
    latent_samples: np.ndarray,
    save_numpy_arrays: bool = True,
    save_grid_image: bool = True,
    save_individual: bool = True,
    max_images_in_grid: int = 64,
    grid_ncols: int = 8,
) -> None:
    cluster_dir = Path(base_output_dir) / f"cluster_{cluster_id}"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    if save_numpy_arrays:
        np.save(cluster_dir / "decoded_images.npy", decoded_images)
        np.save(cluster_dir / "latent_samples.npy", latent_samples)
        np.save(cluster_dir / "decode_label.npy", np.array([decode_label], dtype=np.int32))

    if save_grid_image:
        save_grid(
            images=decoded_images,
            save_path=cluster_dir / "grid.png",
            max_images=max_images_in_grid,
            ncols=grid_ncols,
        )

    if save_individual:
        save_individual_images(decoded_images, cluster_dir)

    print(
        f"Saved cluster {cluster_id} outputs -> "
        f"decoded_images shape: {decoded_images.shape}, "
        f"label used for decoding: {decode_label}"
    )


def main():
    config = DecodeConfig()

    decoder = load_decoder(config.decoder_path)

    cluster_ids = list_available_cluster_ids(config.cluster_sampling_dir)

    base_output_dir = Path(config.cluster_sampling_dir) / config.output_dirname
    if base_output_dir.exists():
        shutil.rmtree(base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print("\nDecoding clusters:")
    for cluster_id in cluster_ids:
        decode_label = load_cluster_label(config.cluster_sampling_dir, cluster_id)
        print(f"  cluster {cluster_id} -> label {decode_label}")

        latent_samples = load_cluster_samples(config.cluster_sampling_dir, cluster_id)

        decoded_images = decode_latent_samples(
            decoder=decoder,
            latent_samples=latent_samples,
            class_label=decode_label,
            num_classes=config.num_classes,
        )

        print(
            f"cluster {cluster_id}: decoded stats -> "
            f"min={decoded_images.min():.4f}, "
            f"max={decoded_images.max():.4f}, "
            f"std={decoded_images.std():.4f}"
        )

        save_cluster_outputs(
            base_output_dir=base_output_dir,
            cluster_id=cluster_id,
            decoded_images=decoded_images,
            decode_label=decode_label,
            latent_samples=latent_samples,
            save_numpy_arrays=config.save_numpy_arrays,
            save_grid_image=config.save_grid,
            save_individual=config.save_individual_images,
            max_images_in_grid=config.max_images_in_grid,
            grid_ncols=config.grid_ncols,
        )

    print("\nDone.")
    print(f"All decoded cluster images saved in: {base_output_dir}")


if __name__ == "__main__":
    main()