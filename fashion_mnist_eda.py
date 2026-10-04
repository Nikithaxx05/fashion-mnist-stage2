"""Requested exploratory data analysis for the Fashion-MNIST development set."""

import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join("work", "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import fashion_mnist


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
FIGURE_DIR = "report_figures"

os.makedirs(FIGURE_DIR, exist_ok=True)

# Use only the 60,000-image development set. The official test set is discarded.
(X_dev_raw, y_dev), _ = fashion_mnist.load_data()

# ------------------------------------------------ class distribution
class_counts = np.bincount(y_dev, minlength=len(CLASS_NAMES))

print("=== Fashion-MNIST development-set class counts ===")
for class_name, count in zip(CLASS_NAMES, class_counts):
    print(f"{class_name}: {count}")

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(CLASS_NAMES, class_counts, color="#4C78A8")
ax.bar_label(bars, padding=3)
ax.set_title("Fashion-MNIST development-set class distribution")
ax.set_xlabel("Class")
ax.set_ylabel("Number of images")
ax.tick_params(axis="x", rotation=35)
ax.set_ylim(0, class_counts.max() * 1.12)
fig.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, "eda_class_distribution.png"), dpi=300)
plt.close(fig)

# ------------------------------------------------ example image grid
# Apply the project's [0,1] normalization assumption for model-ready images.
X_dev_normalized = X_dev_raw.astype("float32") / 255.0

fig, axes = plt.subplots(2, len(CLASS_NAMES), figsize=(16, 4.5))
for class_index, class_name in enumerate(CLASS_NAMES):
    example_indices = np.flatnonzero(y_dev == class_index)[:2]
    for row, example_index in enumerate(example_indices):
        ax = axes[row, class_index]
        ax.imshow(X_dev_normalized[example_index], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        if row == 0:
            ax.set_title(class_name, fontsize=9)

fig.suptitle("Two Fashion-MNIST development examples per class", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(os.path.join(FIGURE_DIR, "eda_class_examples.png"), dpi=300)
plt.close(fig)

# ------------------------------------------------ pixel intensity distribution
pixel_min = int(X_dev_raw.min())
pixel_max = int(X_dev_raw.max())
pixel_mean = float(X_dev_raw.mean())
pixel_std = float(X_dev_raw.std())

print("\n=== Raw development-set pixel statistics ===")
print(f"Min: {pixel_min}")
print(f"Max: {pixel_max}")
print(f"Mean: {pixel_mean:.12f}")
print(f"Standard deviation: {pixel_std:.12f}")

pixel_counts = np.bincount(X_dev_raw.ravel(), minlength=256)
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(np.arange(256), pixel_counts, width=1.0, color="#F58518")
ax.set_title("Fashion-MNIST development-set pixel intensity distribution")
ax.set_xlabel("Raw pixel intensity")
ax.set_ylabel("Number of pixels")
ax.set_xlim(0, 255)
fig.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, "eda_pixel_distribution.png"), dpi=300)
plt.close(fig)
