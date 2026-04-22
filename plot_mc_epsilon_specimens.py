from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ae_utils import _decode, _encode
from train_cvae import load_data


root = Path(__file__).parent.resolve()

dataset_type = "scenairo"
checkpoint_path = root / "runs/scenairo2004/vae_1.pth"
latent_dim = 32
target_classes = [0, 1]
epsilons = list(np.round(np.arange(0.1, 3.0, 0.3), 1)) + [3.0]
n_samples = 5

data = load_data(
    data_csv=None,
    dataset=dataset_type,
    data_root=root / "DATA/ScenAIro",
)

X_train_tensor = data["X_train_tensor"]
y_train = data["y_train"]
input_dim = data["input_dim"]
num_classes = data["class_size"]
image_shape = data["image_shape"]
idx_to_class = {v: k for k, v in data["class_to_idx"].items()}

rows = []
rng = np.random.default_rng(42)

for class_id in target_classes:
    latent = _encode(
        class_id,
        num_classes,
        input_dim,
        latent_dim,
        checkpoint_path,
        dataset_type,
        X_train_tensor,
        y_train,
        image_shape=image_shape,
    )

    center = latent.mean(dim=0)
    radius = torch.linalg.norm(latent - center, dim=1).max().item()

    for eps in epsilons:
        dim = center.shape[0]

        directions = rng.normal(size=(n_samples, dim)).astype(np.float32)
        directions = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12)

        r = rng.random(n_samples).astype(np.float32) ** (1.0 / dim)
        points = center.numpy() + directions * (r[:, None] * eps * radius)
        points = torch.tensor(points, dtype=torch.float32)

        decoded = _decode(
            points,
            class_id,
            num_classes,
            input_dim,
            latent_dim,
            checkpoint_path,
            dataset_type,
            image_shape=image_shape,
        )

        rows.append((idx_to_class[class_id], eps, decoded.cpu()))

fig, axes = plt.subplots(len(rows), n_samples, figsize=(11, 34))

for row_id, row in enumerate(rows):
    class_name, eps, images = row

    for col_id in range(n_samples):
        ax = axes[row_id, col_id]
        ax.imshow(images[col_id].permute(1, 2, 0).numpy())
        ax.axis("off")

        if row_id == 0:
            ax.set_title("sample " + str(col_id + 1), fontsize=8)

        if col_id == 0:
            ax.text(
                -0.08,
                0.5,
                class_name + "\neps=" + str(eps),
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=8,
            )

plt.subplots_adjust(left=0.13, right=0.99, top=0.98, bottom=0.01, hspace=0.55, wspace=0.08)

output_path = root / "monte_carlo_epsilon_specimens.png"
plt.savefig(output_path, dpi=150)
plt.close()

print(output_path)
