from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


@dataclass
class FoolingConfig:
    model_path: str = "artifacts/sut2_classifier.keras"
    decoded_root_dir: str = "runs/scenairo2004/cluster_sampling/decoded_cluster_samples"

    epsilon: float = 0.10
    m_samples_per_cluster: int = 1000

    output_dir: str = "runs/scenairo2004/cluster_sampling/fooling_results"

    expected_height: int = 72
    expected_width: int = 128
    expected_channels: int = 3

    use_range_validity: bool = True
    valid_min: float = 0.0
    valid_max: float = 1.0

    save_per_cluster_predictions: bool = True
    save_fooling_images: bool = True
    save_fooling_pngs: bool = True


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path: str | Path) -> Path:
    return Path(path).resolve()


def load_model(model_path: str | Path) -> tf.keras.Model:
    model_path = resolve_path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from: {model_path}")
    return model


def list_cluster_dirs(decoded_root_dir: str | Path) -> List[Path]:
    root = resolve_path(decoded_root_dir)
    print(f"Looking for decoded clusters in: {root}")

    if not root.exists():
        raise FileNotFoundError(f"Decoded root directory not found: {root}")

    cluster_dirs = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("cluster_")]
    )
    if not cluster_dirs:
        raise FileNotFoundError(f"No cluster folders found inside: {root}")

    print(f"Found {len(cluster_dirs)} cluster folders.")
    return cluster_dirs


def load_cluster_decoded_images(cluster_dir: str | Path) -> np.ndarray:
    path = Path(cluster_dir) / "decoded_images.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing decoded_images.npy in: {cluster_dir}")

    images = np.load(path).astype(np.float32)
    if images.ndim != 4:
        raise ValueError(
            f"Expected decoded images shape (N, H, W, C), got {images.shape} in {cluster_dir}"
        )

    return images


def load_cluster_decode_label(cluster_dir: str | Path) -> int:
    path = Path(cluster_dir) / "decode_label.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing decode_label.npy in: {cluster_dir}")

    label = np.load(path).astype(np.int32).reshape(-1)
    if len(label) == 0:
        raise ValueError(f"decode_label.npy is empty in {cluster_dir}")

    return int(label[0])


def validate_images(images: np.ndarray, config: FoolingConfig) -> np.ndarray:
    if images.shape[1] != config.expected_height:
        raise ValueError(
            f"Expected image height {config.expected_height}, got {images.shape[1]}"
        )
    if images.shape[2] != config.expected_width:
        raise ValueError(
            f"Expected image width {config.expected_width}, got {images.shape[2]}"
        )
    if images.shape[3] != config.expected_channels:
        raise ValueError(
            f"Expected image channels {config.expected_channels}, got {images.shape[3]}"
        )

    if not config.use_range_validity:
        return np.ones((len(images),), dtype=bool)

    valid_mask = (
        (images >= config.valid_min).all(axis=(1, 2, 3))
        & (images <= config.valid_max).all(axis=(1, 2, 3))
    )
    return valid_mask


def predict_labels_and_probs(
    model: tf.keras.Model,
    images: np.ndarray,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    probs = model.predict(images, batch_size=batch_size, verbose=0)
    probs = np.asarray(probs, dtype=np.float32)

    if probs.ndim != 2 or probs.shape[1] != 1:
        raise ValueError(
            f"Expected binary sigmoid probabilities of shape (N, 1), got {probs.shape}"
        )

    positive_probs = probs[:, 0]
    pred_labels = (positive_probs >= 0.5).astype(np.int32)
    pred_conf = np.where(pred_labels == 1, positive_probs, 1.0 - positive_probs).astype(np.float32)

    return pred_labels, pred_conf


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved JSON to: {path}")


def save_csv(rows: List[dict], path: str | Path) -> None:
    if not rows:
        return

    path = Path(path)
    columns = list(rows[0].keys())

    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            f.write(",".join(str(row[col]) for col in columns) + "\n")

    print(f"Saved CSV to: {path}")


def save_latex_table(rows: List[dict], path: str | Path) -> None:
    path = Path(path)
    lines = []

    for row in rows:
        lines.append(
            f"{row['class']} & "
            f"{row['epsilon_ball']:.2f} & "
            f"{row['M']} & "
            f"{row['total_samples']} & "
            f"{row['adversarial']} & "
            f"{row['valid_adv']} & "
            f"{row['asr_percent']:.2f} & "
            f"{row['valid_rate_percent']:.2f} \\\\"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved LaTeX rows to: {path}")


def save_image_as_png(image: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    plt.figure(figsize=(2, 2))
    if image.shape[-1] == 1:
        plt.imshow(image.squeeze(), cmap="gray")
    else:
        plt.imshow(image)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_fooling_images_for_class(
    images: np.ndarray,
    fooling_mask: np.ndarray,
    output_dir: str | Path,
    class_label: int,
    save_pngs: bool = True,
) -> None:
    output_dir = ensure_dir(output_dir)
    fooling_images = images[fooling_mask]

    class_dir = ensure_dir(output_dir / f"class_{class_label}")
    np.save(class_dir / "fooling_images.npy", fooling_images)

    if save_pngs:
        png_dir = ensure_dir(class_dir / "pngs")
        for i, img in enumerate(fooling_images):
            save_image_as_png(img, png_dir / f"adv_{i}.png")

    print(f"Saved class {class_label} fooling images: {fooling_images.shape}")


def save_fooling_images_for_cluster(
    images: np.ndarray,
    fooling_mask: np.ndarray,
    output_dir: str | Path,
    cluster_name: str,
    save_pngs: bool = True,
) -> None:
    output_dir = ensure_dir(output_dir)
    fooling_images = images[fooling_mask]

    cluster_dir = ensure_dir(output_dir / cluster_name)
    np.save(cluster_dir / "fooling_images.npy", fooling_images)

    if save_pngs:
        png_dir = ensure_dir(cluster_dir / "pngs")
        for i, img in enumerate(fooling_images):
            save_image_as_png(img, png_dir / f"adv_{i}.png")

    print(f"Saved {cluster_name} fooling images: {fooling_images.shape}")


def print_results_table(rows: List[dict]) -> None:
    if not rows:
        print("No results to display.")
        return

    headers = [
        "Class",
        "epsilon-ball",
        "M",
        "Total Samples",
        "Adversarial",
        "Valid Adv.",
        "ASR (%)",
        "Valid Rate (%)",
    ]

    table_rows = []
    for row in rows:
        table_rows.append([
            str(row["class"]),
            f"{row['epsilon_ball']:.2f}",
            str(row["M"]),
            str(row["total_samples"]),
            str(row["adversarial"]),
            str(row["valid_adv"]),
            f"{row['asr_percent']:.2f}",
            f"{row['valid_rate_percent']:.2f}",
        ])

    col_widths = []
    for i, header in enumerate(headers):
        max_width = len(header)
        for r in table_rows:
            max_width = max(max_width, len(r[i]))
        col_widths.append(max_width)

    def format_row(values: List[str]) -> str:
        return " | ".join(v.ljust(col_widths[i]) for i, v in enumerate(values))

    separator = "-+-".join("-" * w for w in col_widths)

    print("\nResults table:")
    print(format_row(headers))
    print(separator)
    for r in table_rows:
        print(format_row(r))


def run_fooling_evaluation(config: FoolingConfig) -> List[dict]:
    print("Starting fooling evaluation...")
    print("Model path:", resolve_path(config.model_path))
    print("Decoded root dir:", resolve_path(config.decoded_root_dir))
    print("Output dir:", resolve_path(config.output_dir))

    output_dir = ensure_dir(config.output_dir)
    model = load_model(config.model_path)
    cluster_dirs = list_cluster_dirs(config.decoded_root_dir)

    per_class_data: Dict[int, dict] = {}

    fooling_by_cluster_dir = ensure_dir(output_dir / "fooling_by_cluster")
    fooling_by_class_dir = ensure_dir(output_dir / "fooling_by_class")

    for cluster_dir in cluster_dirs:
        print(f"Processing {cluster_dir.name} ...")

        images = load_cluster_decoded_images(cluster_dir)
        expected_label = load_cluster_decode_label(cluster_dir)

        valid_mask = validate_images(images, config)
        pred_labels, pred_conf = predict_labels_and_probs(model, images)

        adversarial_mask = pred_labels != expected_label
        valid_adv_mask = adversarial_mask & valid_mask

        save_fooling_images_for_cluster(
            images=images,
            fooling_mask=valid_adv_mask,
            output_dir=fooling_by_cluster_dir,
            cluster_name=cluster_dir.name,
            save_pngs=config.save_fooling_pngs,
        )

        if expected_label not in per_class_data:
            per_class_data[expected_label] = {
                "images": [],
                "valid_mask": [],
                "adversarial_mask": [],
                "valid_adv_mask": [],
                "pred_labels": [],
                "pred_conf": [],
            }

        per_class_data[expected_label]["images"].append(images)
        per_class_data[expected_label]["valid_mask"].append(valid_mask)
        per_class_data[expected_label]["adversarial_mask"].append(adversarial_mask)
        per_class_data[expected_label]["valid_adv_mask"].append(valid_adv_mask)
        per_class_data[expected_label]["pred_labels"].append(pred_labels)
        per_class_data[expected_label]["pred_conf"].append(pred_conf)

        if config.save_per_cluster_predictions:
            cluster_out = ensure_dir(output_dir / cluster_dir.name)
            cluster_rows = []
            for i in range(len(images)):
                cluster_rows.append(
                    {
                        "sample_index": i,
                        "expected_label": expected_label,
                        "pred_label": int(pred_labels[i]),
                        "pred_confidence": float(pred_conf[i]),
                        "is_valid": int(valid_mask[i]),
                        "is_adversarial": int(adversarial_mask[i]),
                        "is_valid_adversarial": int(valid_adv_mask[i]),
                    }
                )
            save_csv(cluster_rows, cluster_out / "predictions.csv")

    summary_rows: List[dict] = []

    for class_label in sorted(per_class_data.keys()):
        class_images = np.concatenate(per_class_data[class_label]["images"], axis=0)
        class_valid_mask = np.concatenate(per_class_data[class_label]["valid_mask"], axis=0)
        class_adv_mask = np.concatenate(per_class_data[class_label]["adversarial_mask"], axis=0)
        class_valid_adv_mask = np.concatenate(per_class_data[class_label]["valid_adv_mask"], axis=0)

        total_samples = int(len(class_images))
        adversarial = int(class_adv_mask.sum())
        valid_adv = int(class_valid_adv_mask.sum())
        valid_count = int(class_valid_mask.sum())

        asr_percent = 100.0 * valid_adv / total_samples if total_samples > 0 else 0.0
        valid_rate_percent = 100.0 * valid_count / total_samples if total_samples > 0 else 0.0

        row = {
            "class": class_label,
            "epsilon_ball": float(config.epsilon),
            "M": int(config.m_samples_per_cluster),
            "total_samples": total_samples,
            "adversarial": adversarial,
            "valid_adv": valid_adv,
            "asr_percent": round(asr_percent, 2),
            "valid_rate_percent": round(valid_rate_percent, 2),
        }
        summary_rows.append(row)

        if config.save_fooling_images:
            save_fooling_images_for_class(
                images=class_images,
                fooling_mask=class_valid_adv_mask,
                output_dir=fooling_by_class_dir,
                class_label=class_label,
                save_pngs=config.save_fooling_pngs,
            )

    save_json(asdict(config), output_dir / "fooling_config.json")
    save_json({"rows": summary_rows}, output_dir / "fooling_summary.json")
    save_csv(summary_rows, output_dir / "fooling_results.csv")
    save_latex_table(summary_rows, output_dir / "fooling_results_latex.txt")

    print_results_table(summary_rows)

    return summary_rows


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    base_dir = repo_root / "runs/scenairo2004/cluster_sampling"
    epsilon_dirs = sorted(base_dir.glob("epsilon_*"))

    if not epsilon_dirs:
        raise FileNotFoundError(f"No epsilon_* folders found inside: {base_dir.resolve()}")

    for epsilon_dir in epsilon_dirs:
        epsilon = float(epsilon_dir.name.removeprefix("epsilon_"))
        config = FoolingConfig(
            model_path=str(repo_root / "artifacts/sut2_classifier.keras"),
            decoded_root_dir=str(epsilon_dir / "decoded_cluster_samples"),
            epsilon=epsilon,
            m_samples_per_cluster=1000,
            output_dir=str(epsilon_dir / "fooling_results"),
            expected_height=72,
            expected_width=128,
            expected_channels=3,
            use_range_validity=True,
            valid_min=0.0,
            valid_max=1.0,
            save_per_cluster_predictions=True,
            save_fooling_images=True,
            save_fooling_pngs=True,
        )

        run_fooling_evaluation(config)


if __name__ == "__main__":
    main()
