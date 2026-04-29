from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.utils import to_categorical


@dataclass
class TrainConfig:
    image_size: Tuple[int, int] = (28, 28)
    batch_size: int = 128
    epochs: int = 10
    seed: int = 42
    learning_rate: float = 1e-3
    valid_size: float = 0.1
    num_classes: int = 10

    artifacts_dir: str = r"C:\Users\akhiat\Desktop\Hackathon\ASAB\artifacts"
    keras_model_name: str = "sut1_mnist_classifier.keras"
    saved_model_dirname: str = "sut1_mnist_saved_model"
    onnx_model_name: str = "sut1_mnist_classifier.onnx"

    export_saved_model: bool = True
    export_onnx: bool = True
    verify_onnx_inference: bool = True
    onnx_opset: int = 13


def set_global_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_mnist_data(config: TrainConfig):
    """
    Loads MNIST and creates train / validation / test sets.

    Returns
    -------
    x_train, y_train, x_val, y_val, x_test, y_test
    """
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize to [0, 1]
    x_train_full = x_train_full.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # Add channel dimension: (N, 28, 28) -> (N, 28, 28, 1)
    x_train_full = np.expand_dims(x_train_full, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # Optional resize if image_size != (28, 28)
    if config.image_size != (28, 28):
        x_train_full = tf.image.resize(x_train_full, config.image_size).numpy()
        x_test = tf.image.resize(x_test, config.image_size).numpy()

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=config.valid_size,
        random_state=config.seed,
        shuffle=True,
        stratify=y_train_full,
    )

    # One-hot encoding for multiclass classification
    y_train = to_categorical(y_train, num_classes=config.num_classes)
    y_val = to_categorical(y_val, num_classes=config.num_classes)
    y_test = to_categorical(y_test, num_classes=config.num_classes)

    return x_train, y_train, x_val, y_val, x_test, y_test


def print_data_info(x_train, y_train, x_val, y_val, x_test, y_test) -> None:
    print("\nDataset summary:")
    print(f"  Training samples   : {len(x_train)}")
    print(f"  Validation samples : {len(x_val)}")
    print(f"  Test samples       : {len(x_test)}")
    print(f"  Input shape        : {x_train.shape[1:]}")
    print(f"  Number of classes  : {y_train.shape[1]}")


def build_mnist_cnn(
    input_shape: Tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 10,
    learning_rate: float = 1e-4,
) -> tf.keras.Model:
    model = Sequential(
        [
            Input(shape=input_shape),

            Conv2D(8, (3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(16, (5, 5), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2)),

            #Conv2D(32, (5, 5), activation="relu", padding="same"),
            #MaxPooling2D(pool_size=(2, 2)),

            Flatten(),

            Dense(100, activation="relu"),
            Dropout(0.35),
            Dense(20, activation="relu"),
            Dropout(0.2),

            Dense(num_classes, activation="softmax"),
        ]
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    epochs: int,
):
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
    )
    return history


def evaluate_model(model: tf.keras.Model, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
    }


def plot_training_history(history) -> None:
    history_dict = history.history
    epochs = range(1, len(history_dict["loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history_dict["loss"], label="Training loss")
    plt.plot(epochs, history_dict["val_loss"], label="Validation loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if "accuracy" in history_dict and "val_accuracy" in history_dict:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history_dict["accuracy"], label="Training accuracy")
        plt.plot(epochs, history_dict["val_accuracy"], label="Validation accuracy")
        plt.title("Accuracy Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


def ensure_artifacts_dir(path: str) -> Path:
    artifacts_dir = Path(path)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir


def save_keras_model(model: tf.keras.Model, artifacts_dir: Path, file_name: str) -> Path:
    model_path = artifacts_dir / file_name
    model.save(model_path)
    print(f"\nSaved Keras model to: {model_path}")
    return model_path


def export_saved_model(model: tf.keras.Model, artifacts_dir: Path, dir_name: str) -> Path:
    export_path = artifacts_dir / dir_name

    if export_path.exists():
        shutil.rmtree(export_path)

    model.export(str(export_path))
    print(f"Exported TensorFlow SavedModel to: {export_path}")
    return export_path


def export_saved_model_to_onnx(
    saved_model_path: Path,
    onnx_output_path: Path,
    opset: int = 13,
) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "tf2onnx.convert",
        "--saved-model",
        str(saved_model_path),
        "--output",
        str(onnx_output_path),
        "--opset",
        str(opset),
    ]

    print("\nRunning ONNX export command:")
    print(" ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "Could not run tf2onnx. Install it with: python -m pip install -U tf2onnx onnx onnxruntime"
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "tf2onnx conversion failed. Check the log printed above."
        ) from e

    print(f"Exported ONNX model to: {onnx_output_path}")
    return onnx_output_path


def load_keras_model_for_inference(model_path: str | Path) -> tf.keras.Model:
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded Keras model from: {model_path}")
    return model


def predict_with_keras(model: tf.keras.Model, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return model.predict(x, verbose=0)


def predict_with_onnx(onnx_model_path: str | Path, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    x = np.asarray(x, dtype=np.float32)
    session = ort.InferenceSession(str(onnx_model_path))
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: x})
    return outputs[0]


def verify_onnx_against_keras(
    keras_model: tf.keras.Model,
    onnx_model_path: Path,
    sample_batch: np.ndarray,
    tolerance: float = 1e-4,
) -> dict:
    keras_preds = predict_with_keras(keras_model, sample_batch)
    onnx_preds = predict_with_onnx(onnx_model_path, sample_batch)

    max_abs_diff = float(np.max(np.abs(keras_preds - onnx_preds)))
    mean_abs_diff = float(np.mean(np.abs(keras_preds - onnx_preds)))
    is_close = bool(np.allclose(keras_preds, onnx_preds, atol=tolerance))

    results = {
        "keras_shape": tuple(keras_preds.shape),
        "onnx_shape": tuple(onnx_preds.shape),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "allclose": is_close,
        "tolerance": tolerance,
    }

    print("\nONNX verification:")
    for key, value in results.items():
        print(f"  {key}: {value}")

    return results


def run_training_pipeline(config: TrainConfig):
    set_global_seed(config.seed)

    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist_data(config)
    print_data_info(x_train, y_train, x_val, y_val, x_test, y_test)

    model = build_mnist_cnn(
        input_shape=(config.image_size[0], config.image_size[1], 1),
        num_classes=config.num_classes,
        learning_rate=config.learning_rate,
    )

    model.summary()

    history = train_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        batch_size=config.batch_size,
        epochs=config.epochs,
    )

    plot_training_history(history)

    test_results = evaluate_model(model, x_test, y_test)

    print("\nTest results:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")

    artifacts_dir = ensure_artifacts_dir(config.artifacts_dir)

    keras_model_path = save_keras_model(
        model=model,
        artifacts_dir=artifacts_dir,
        file_name=config.keras_model_name,
    )

    saved_model_path: Optional[Path] = None
    if config.export_saved_model or config.export_onnx:
        saved_model_path = export_saved_model(
            model=model,
            artifacts_dir=artifacts_dir,
            dir_name=config.saved_model_dirname,
        )

    onnx_model_path: Optional[Path] = None
    onnx_verification: Optional[dict] = None

    if config.export_onnx:
        if saved_model_path is None:
            raise RuntimeError("SavedModel export is required before ONNX conversion.")

        onnx_model_path = artifacts_dir / config.onnx_model_name
        export_saved_model_to_onnx(
            saved_model_path=saved_model_path,
            onnx_output_path=onnx_model_path,
            opset=config.onnx_opset,
        )

        if config.verify_onnx_inference:
            sample_batch = x_test[:8].astype(np.float32)
            onnx_verification = verify_onnx_against_keras(
                keras_model=model,
                onnx_model_path=onnx_model_path,
                sample_batch=sample_batch,
                tolerance=1e-4,
            )

    return {
        "model": model,
        "history": history,
        "test_results": test_results,
        "keras_model_path": keras_model_path,
        "saved_model_path": saved_model_path,
        "onnx_model_path": onnx_model_path,
        "onnx_verification": onnx_verification,
    }


if __name__ == "__main__":
    config = TrainConfig(
        image_size=(28, 28),
        batch_size=64,
        epochs=20,
        seed=42,
        learning_rate=1e-3,
        valid_size=0.2,
        num_classes=10,
        artifacts_dir=r"C:\Users\akhiat\Desktop\Hackathon\ASAB\artifacts",
        export_saved_model=True,
        export_onnx=True,
        verify_onnx_inference=True,
        onnx_opset=13,
    )

    outputs = run_training_pipeline(config)