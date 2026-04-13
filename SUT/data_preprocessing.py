from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


AUTOTUNE = tf.data.AUTOTUNE


@dataclass
class DataConfig:
    dataset: str = "mnist"
    single_class: Optional[int] = None
    valid_size: float = 0.1
    random_state: int = 42
    image_size: Optional[Tuple[int, int]] = None   # e.g. (32, 32)
    batch_size: int = 64
    shuffle_buffer: int = 1000


def _validate_config(config: DataConfig) -> None:
    if config.dataset.lower() != "mnist":
        raise ValueError("This preprocessing file only supports dataset='mnist'.")
    if not (0.0 < config.valid_size < 1.0):
        raise ValueError("valid_size must be in (0, 1).")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be > 0.")


def _preprocess_mnist_image(
    image: tf.Tensor,
    label: tf.Tensor,
    image_size: Optional[Tuple[int, int]],
) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.cast(image, tf.float32)

    # MNIST comes as [H, W], so add channel dimension -> [H, W, 1]
    if len(image.shape) == 2:
        image = tf.expand_dims(image, axis=-1)

    # Normalize to [0, 1]
    image = image / 255.0

    # Optional resize
    if image_size is not None:
        image = tf.image.resize(image, image_size)

    return image, tf.cast(label, tf.int32)


def _build_tf_dataset(
    X: np.ndarray,
    y: np.ndarray,
    config: DataConfig,
    training: bool,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X, y))

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(X), config.shuffle_buffer),
            seed=config.random_state,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(
        lambda x, y: _preprocess_mnist_image(x, y, config.image_size),
        num_parallel_calls=AUTOTUNE,
    )

    ds = ds.batch(config.batch_size).prefetch(AUTOTUNE)
    return ds


def prepare_mnist_data(config: DataConfig):
    """
    TensorFlow preprocessing pipeline for MNIST.

    Returns
    -------
    train_ds : tf.data.Dataset
        Dataset yielding (images, labels) batches.
    valid_ds : tf.data.Dataset
        Dataset yielding (images, labels) batches.
    metadata : dict
        Information about shapes, classes, and split sizes.
    """
    _validate_config(config)

    (X_full, y_full), _ = tf.keras.datasets.mnist.load_data()

    X = X_full
    y = y_full

    # Optional filtering to a single class
    if config.single_class is not None:
        mask = (y == config.single_class)
        X = X[mask]
        y = y[mask]

        if len(X) == 0:
            raise ValueError(f"No MNIST samples found for class {config.single_class}.")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=config.valid_size,
        random_state=config.random_state,
        shuffle=True,
        stratify=y if config.single_class is None else None,
    )

    train_ds = _build_tf_dataset(X_train, y_train, config, training=True)
    valid_ds = _build_tf_dataset(X_valid, y_valid, config, training=False)

    input_shape = (
        (config.image_size[0], config.image_size[1], 1)
        if config.image_size is not None
        else (28, 28, 1)
    )

    metadata = {
        "dataset": "mnist",
        "input_shape": input_shape,
        "n_classes": 1 if config.single_class is not None else 10,
        "train_size": len(X_train),
        "valid_size": len(X_valid),
        "single_class": config.single_class,
    }

    return train_ds, valid_ds, metadata