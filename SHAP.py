#!/usr/bin/env python3
"""
SHAP magnitude overlay — single‑hue green transparency.
"""
import os, numpy as np, shap
import matplotlib.pyplot as plt
from PIL import Image as PILImage

import util
from model import MultiLogisticRegressionModel


# 1.  Train a very small logistic model on MNIST
train_ds  = util.get_dataset("mnist_train")
X_train   = np.array(train_ds.xs).reshape(-1, 28 * 28).astype(float)

model = MultiLogisticRegressionModel(28 * 28, 10, learning_rate=1e-2)
model.train(train_ds)


def predict(flat_imgs: np.ndarray) -> np.ndarray:
    imgs = flat_imgs.reshape(-1, 28, 28)
    return np.array([model.hypothesis(img) for img in imgs])


# 2.  SHAP explainer with a tiny background
bg = X_train[np.random.choice(len(X_train), 100, replace=False)]
explainer = shap.Explainer(predict, bg)


# 3.  Load ten custom images we want to explain
def load_gray(path: str) -> np.ndarray:
    return np.array(PILImage.open(path).convert('L')).astype(float) / 255.0

imgs = np.stack([
    load_gray(os.path.join("mnist_png", "explained_imgs", f"{i}.png"))
    for i in range(10)
])                                # (10,28,28)
flat_imgs = imgs.reshape(10, -1)


# 4.  Explain once
exp       = explainer(flat_imgs, max_evals=2000)
probs     = predict(flat_imgs)
pred_cls  = np.argmax(probs, axis=1)


# 6.  Plot originals vs green‑magnitude overlays
fig, ax = plt.subplots(10, 2, figsize=(6, 55))
fig.subplots_adjust(hspace=1.0)

for i in range(10):
    imp_map = np.abs(exp.values[i, :, pred_cls[i]]).reshape(28, 28)

    # left: original
    ax[i, 0].imshow(imgs[i], cmap='gray')
    ax[i, 0].set_title(f"Digit {i}", fontsize=12)
    ax[i, 0].axis('off')

    # right: overlay
    ax[i, 1].imshow(imp_map)
    ax[i, 1].set_title(
        f"Pred {pred_cls[i]}  •  Conf {probs[i, pred_cls[i]]:.2f}", fontsize=12
    )
    ax[i, 1].axis('off')

plt.tight_layout()
plt.show()
