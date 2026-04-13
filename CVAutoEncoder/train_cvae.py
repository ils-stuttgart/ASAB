from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


@dataclass
class CVAEConfig:
    model_name: str = "cvae_mnist"
    model_dir: str = "./runs"
    epochs: int = 30
    batch_size: int = 64
    latent_dim: int = 10
    num_classes: int = 10
    image_shape: Tuple[int, int, int] = (28, 28, 1)
    learning_rate: float = 1e-4
    seed: int = 42
    valid_split: float = 0.2
    beta: float = 2.0


def set_global_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def prepare_mnist_data(batch_size: int, valid_split: float = 0.1):
    (x_full, y_full), _ = tf.keras.datasets.mnist.load_data()

    x_full = x_full.astype(np.float32) / 255.0
    x_full = np.expand_dims(x_full, axis=-1)

    y_full = y_full.astype(np.int32)
    y_full_oh = tf.keras.utils.to_categorical(y_full, num_classes=10).astype(np.float32)

    n_total = len(x_full)
    n_valid = int(valid_split * n_total)

    x_valid = x_full[:n_valid]
    y_valid = y_full_oh[:n_valid]

    x_train = x_full[n_valid:]
    y_train = y_full_oh[n_valid:]

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(10000, seed=42).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    valid_ds = valid_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, valid_ds, x_train, y_train, x_valid, y_valid


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs, training=None):
        mu, log_var = inputs
        eps = tf.random.normal(shape=tf.shape(mu))
        std = tf.exp(0.5 * log_var)
        return mu + std * eps


def build_encoder(latent_dim: int, num_classes: int, image_shape=(28, 28, 1)) -> tf.keras.Model:
    image_input = tf.keras.Input(shape=image_shape, name="image_input")
    label_input = tf.keras.Input(shape=(num_classes,), name="label_input")

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(image_input)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)

    x = tf.keras.layers.Concatenate()([x, label_input])
    x = tf.keras.layers.Dense(64, activation="relu")(x)

    mu = tf.keras.layers.Dense(latent_dim, name="mu")(x)
    log_var = tf.keras.layers.Dense(latent_dim, name="log_var")(x)
    z = Sampling(name="sampling")([mu, log_var])

    return tf.keras.Model(
        inputs=[image_input, label_input],
        outputs=[mu, log_var, z],
        name="encoder",
    )


def build_decoder(latent_dim: int, num_classes: int, image_shape=(28, 28, 1)) -> tf.keras.Model:
    latent_input = tf.keras.Input(shape=(latent_dim,), name="latent_input")
    label_input = tf.keras.Input(shape=(num_classes,), name="label_input")

    flat_dim = int(np.prod(image_shape))

    x = tf.keras.layers.Concatenate()([latent_input, label_input])
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(flat_dim, activation="sigmoid")(x)

    output = tf.keras.layers.Reshape(image_shape, name="reconstruction")(x)

    return tf.keras.Model(
        inputs=[latent_input, label_input],
        outputs=output,
        name="decoder",
    )


class CVAE(tf.keras.Model):
    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        image_shape=(28, 28, 1),
        beta: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.image_shape = image_shape
        self.beta = beta

        self.encoder = build_encoder(latent_dim, num_classes, image_shape)
        self.decoder = build_decoder(latent_dim, num_classes, image_shape)

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs, training=None):
        x, c = inputs
        mu, log_var, z = self.encoder([x, c], training=training)
        recon = self.decoder([z, c], training=training)
        return recon, mu, log_var

    def compute_losses(self, x, c, training=False):
        mu, log_var, z = self.encoder([x, c], training=training)
        recon = self.decoder([z, c], training=training)

        recon_loss_per_sample = tf.reduce_sum(tf.square(x - recon), axis=[1, 2, 3])
        recon_loss = tf.reduce_mean(recon_loss_per_sample)

        kl_per_sample = -0.5 * tf.reduce_sum(
            1.0 + log_var - tf.square(mu) - tf.exp(log_var),
            axis=1,
        )
        kl_loss = tf.reduce_mean(kl_per_sample)

        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss, recon, mu, log_var

    def train_step(self, data):
        x, c = data

        with tf.GradientTape() as tape:
            total_loss, recon_loss, kl_loss, _, _, _ = self.compute_losses(
                x, c, training=True
            )

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        x, c = data
        total_loss, recon_loss, kl_loss, _, _, _ = self.compute_losses(
            x, c, training=False
        )

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def encode(self, x, c):
        return self.encoder([x, c], training=False)

    def decode(self, z, c):
        return self.decoder([z, c], training=False)


def plot_training_history(history, save_path: str | Path) -> None:
    history_dict = history.history
    epochs = range(1, len(history_dict["loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history_dict["loss"], label="Train total loss")
    if "val_loss" in history_dict:
        plt.plot(epochs, history_dict["val_loss"], label="Val total loss")
    if "reconstruction_loss" in history_dict:
        plt.plot(epochs, history_dict["reconstruction_loss"], label="Train recon loss")
    if "kl_loss" in history_dict:
        plt.plot(epochs, history_dict["kl_loss"], label="Train KL loss")
    plt.title("CVAE Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_reconstructions(
    model: CVAE,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    save_path: str | Path,
    n: int = 10,
):
    x_batch = x_valid[:n]
    c_batch = y_valid[:n]
    recon, _, _ = model([x_batch, c_batch], training=False)
    recon = recon.numpy()

    fig, axes = plt.subplots(2, n, figsize=(1.5 * n, 3))
    for i in range(n):
        axes[0, i].imshow(x_batch[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Recon")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_reconstruction_error(
    model: CVAE,
    x_data: np.ndarray,
    y_data: np.ndarray,
    save_path: str | Path,
    bins: int = 50,
) -> np.ndarray:
    recon, _, _ = model([x_data, y_data], training=False)
    recon = recon.numpy()

    errors = np.mean((x_data - recon) ** 2, axis=(1, 2, 3))

    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=bins)
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("Per-sample MSE")
    plt.ylabel("Number of samples")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return errors


def plot_reconstruction_error_curve(
    errors: np.ndarray,
    save_path: str | Path,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(errors)
    plt.title("Reconstruction Error per Sample")
    plt.xlabel("Sample index")
    plt.ylabel("Per-sample MSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_reconstruction_errors(errors: np.ndarray, save_path: str | Path) -> None:
    np.save(save_path, errors)


def sample_conditioned_digits(
    model: CVAE,
    latent_dim: int,
    num_classes: int,
    save_path: str | Path,
    n_per_class: int = 10,
):
    fig, axes = plt.subplots(num_classes, n_per_class, figsize=(n_per_class, num_classes))

    for cls in range(num_classes):
        c = tf.one_hot([cls] * n_per_class, depth=num_classes, dtype=tf.float32)
        z = tf.random.normal(shape=(n_per_class, latent_dim))
        samples = model.decode(z, c).numpy()

        for j in range(n_per_class):
            axes[cls, j].imshow(samples[j].squeeze(), cmap="gray")
            axes[cls, j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_latent_vectors_for_dataset(
    model: CVAE,
    x_data: np.ndarray,
    y_data: np.ndarray,
    output_dir: str | Path,
    prefix: str = "valid",
) -> None:
    """
    Save one latent vector per input sample.

    Saves:
      - {prefix}_latent_mu.npy
      - {prefix}_latent_log_var.npy
      - {prefix}_latent_z.npy
      - {prefix}_labels.npy
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mu, log_var, z = model.encode(x_data, y_data)

    mu = mu.numpy()
    log_var = log_var.numpy()
    z = z.numpy()

    labels = np.argmax(y_data, axis=1).astype(np.int32)

    np.save(output_dir / f"{prefix}_latent_mu.npy", mu)
    np.save(output_dir / f"{prefix}_latent_log_var.npy", log_var)
    np.save(output_dir / f"{prefix}_latent_z.npy", z)
    np.save(output_dir / f"{prefix}_labels.npy", labels)

    print(f"Saved latent vectors for prefix='{prefix}'")
    print(f"  mu shape      : {mu.shape}")
    print(f"  log_var shape : {log_var.shape}")
    print(f"  z shape       : {z.shape}")
    print(f"  labels shape  : {labels.shape}")


def main():
    config = CVAEConfig()
    set_global_seed(config.seed)

    train_ds, valid_ds, x_train, y_train, x_valid, y_valid = prepare_mnist_data(
        config.batch_size,
        config.valid_split,
    )

    model = CVAE(
        latent_dim=config.latent_dim,
        num_classes=config.num_classes,
        image_shape=config.image_shape,
        beta=config.beta,
        name="cvae_mnist",
    )

    dummy_x = tf.zeros((1, *config.image_shape), dtype=tf.float32)
    dummy_c = tf.zeros((1, config.num_classes), dtype=tf.float32)
    _ = model([dummy_x, dummy_c], training=False)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate))

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=config.epochs,
        verbose=1,
    )

    output_dir = Path(config.model_dir) / config.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_weights(output_dir / "cvae.weights.h5")
    model.encoder.save(output_dir / "encoder.keras")
    model.decoder.save(output_dir / "decoder.keras")

    plot_training_history(history, output_dir / "cvae_loss.png")

    save_reconstructions(
        model,
        x_valid,
        y_valid,
        output_dir / "reconstructions.png",
        n=10,
    )

    sample_conditioned_digits(
        model,
        latent_dim=config.latent_dim,
        num_classes=config.num_classes,
        save_path=output_dir / "conditional_samples.png",
        n_per_class=10,
    )

    recon_errors = plot_reconstruction_error(
        model,
        x_valid,
        y_valid,
        output_dir / "reconstruction_error_hist.png",
        bins=50,
    )

    plot_reconstruction_error_curve(
        recon_errors,
        output_dir / "reconstruction_error_curve.png",
    )

    save_reconstruction_errors(
        recon_errors,
        output_dir / "reconstruction_errors.npy",
    )

    save_latent_vectors_for_dataset(
        model,
        x_train,
        y_train,
        output_dir,
        prefix="train",
    )

    save_latent_vectors_for_dataset(
        model,
        x_valid,
        y_valid,
        output_dir,
        prefix="valid",
    )

    print(
        f"Reconstruction error stats -> "
        f"mean: {recon_errors.mean():.6f}, "
        f"std: {recon_errors.std():.6f}, "
        f"min: {recon_errors.min():.6f}, "
        f"max: {recon_errors.max():.6f}"
    )

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()