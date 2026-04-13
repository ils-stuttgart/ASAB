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
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator


@dataclass
class TrainConfig:
    data_root: str = r"C:\Users\akhiat\Desktop\Hackathon\ASAB\DATA\ScenAIro"
    image_size: Tuple[int, int] = (100, 100)
    batch_size: int = 64
    epochs: int = 10
    seed: int = 42
    learning_rate: float = 1e-3

    train_dirname: str = "train"
    val_dirname: str = "val"
    test_dirname: str = "test"
    class_mode: str = "binary"

    artifacts_dir: str = r"C:\Users\akhiat\Desktop\Hackathon\ASAB\artifacts"
    keras_model_name: str = "sut2_classifier.keras"
    saved_model_dirname: str = "sut2_saved_model"
    onnx_model_name: str = "sut2_classifier.onnx"

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


def verify_dataset_structure(data_root: str) -> None:
    """
    Expected structure:
        data_root/
            train/
                norunway/
                runway/
            val/
                norunway/
                runway/
            test/
                norunway/
                runway/
    """
    root = Path(data_root)

    expected_dirs = [
        root / "train" / "norunway",
        root / "train" / "runway",
        root / "val" / "norunway",
        root / "val" / "runway",
        root / "test" / "norunway",
        root / "test" / "runway",
    ]

    missing = [str(p) for p in expected_dirs if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "The following required dataset folders are missing:\n" + "\n".join(missing)
        )

    print("Dataset structure verified successfully.")


def build_data_generators(config: TrainConfig):
    verify_dataset_structure(config.data_root)

    data_root = Path(config.data_root)
    train_dir = data_root / config.train_dirname
    val_dir = data_root / config.val_dirname
    test_dir = data_root / config.test_dirname

    datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = datagen.flow_from_directory(
        directory=str(train_dir),
        target_size=config.image_size,
        batch_size=config.batch_size,
        shuffle=True,
        class_mode=config.class_mode,
    )

    validation_generator = datagen.flow_from_directory(
        directory=str(val_dir),
        target_size=config.image_size,
        batch_size=config.batch_size,
        shuffle=False,
        class_mode=config.class_mode,
    )

    test_generator = datagen.flow_from_directory(
        directory=str(test_dir),
        target_size=config.image_size,
        batch_size=config.batch_size,
        shuffle=False,
        class_mode=config.class_mode,
    )

    return train_generator, validation_generator, test_generator


def print_generator_info(train_generator, validation_generator, test_generator) -> None:
    print("\nDataset summary:")
    print(f"  Training samples   : {train_generator.samples}")
    print(f"  Validation samples : {validation_generator.samples}")
    print(f"  Test samples       : {test_generator.samples}")
    print(f"  Class indices      : {train_generator.class_indices}")
    print(f"  Input batch shape  : {train_generator.image_shape}")


def build_binary_cnn(
    input_shape: Tuple[int, int, int] = (100, 100, 3),
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    model = Sequential(
        [
            Input(shape=input_shape),

            Conv2D(filters=16, kernel_size=(3, 3), padding="same", strides=1),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2), strides=2),

            Conv2D(filters=32, kernel_size=(5, 5), padding="valid", strides=1),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2), strides=2),

            Flatten(),

            Dense(200),
            Activation("relu"),
            Dropout(0.3),

            Dense(100),
            Activation("relu"),

            Dense(1),
            Activation("sigmoid"),
        ]
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    return model


def train_model(
    model: tf.keras.Model,
    train_generator,
    validation_generator,
    epochs: int,
):
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        verbose=1,
    )
    return history


def evaluate_model(model: tf.keras.Model, test_generator) -> dict:
    loss, accuracy = model.evaluate(test_generator, verbose=0)
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
    """
    Save the full Keras model for later TensorFlow/Keras inference.
    """
    model_path = artifacts_dir / file_name
    model.save(model_path)
    print(f"\nSaved Keras model to: {model_path}")
    return model_path


def export_saved_model(model: tf.keras.Model, artifacts_dir: Path, dir_name: str) -> Path:
    """
    Export a TensorFlow SavedModel for serving / conversion.
    """
    export_path = artifacts_dir / dir_name

    if export_path.exists():
        shutil.rmtree(export_path)

    # Keras export() creates an inference-focused SavedModel
    model.export(str(export_path))
    print(f"Exported TensorFlow SavedModel to: {export_path}")
    return export_path


def export_saved_model_to_onnx(
    saved_model_path: Path,
    onnx_output_path: Path,
    opset: int = 13,
) -> Path:
    """
    Convert a TensorFlow SavedModel to ONNX using tf2onnx CLI.
    This is usually more reliable than converting directly from .keras.
    """
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
            "Could not run tf2onnx. Install it in the active environment with: "
            "pip install tf2onnx onnx onnxruntime"
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "tf2onnx conversion failed. This usually means a version mismatch or "
            "an unsupported conversion path. The safer route is SavedModel -> ONNX, "
            "which this script already uses. Check the tf2onnx log printed above."
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

    train_generator, validation_generator, test_generator = build_data_generators(config)
    print_generator_info(train_generator, validation_generator, test_generator)

    model = build_binary_cnn(
        input_shape=(config.image_size[0], config.image_size[1], 3),
        learning_rate=config.learning_rate,
    )

    model.summary()

    history = train_model(
        model=model,
        train_generator=train_generator,
        validation_generator=validation_generator,
        epochs=config.epochs,
    )

    plot_training_history(history)

    test_results = evaluate_model(model, test_generator)

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
            x_batch, _ = next(test_generator)
            sample_batch = x_batch[: min(8, len(x_batch))].astype(np.float32)
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
        data_root=r"C:\Users\akhiat\Desktop\Hackathon\ASAB\DATA\ScenAIro",
        image_size=(100, 100),
        batch_size=64,
        epochs=2,
        seed=42,
        learning_rate=1e-3,
        artifacts_dir=r"C:\Users\akhiat\Desktop\Hackathon\ASAB\artifacts",
        export_saved_model=True,
        export_onnx=True,
        verify_onnx_inference=True,
        onnx_opset=13,
    )

    outputs = run_training_pipeline(config)