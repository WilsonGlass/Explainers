import os

import numpy as np
import shap
import util
from model import MultiLogisticRegressionModel
from PIL import Image as PILImage
import matplotlib.pyplot as plt

train_data = util.get_dataset("mnist_train")
test_data = util.get_dataset("mnist_test")
train_arrays = np.array(train_data.xs).reshape(-1, 28*28).astype(float)
test_arrays = np.array(test_data.xs).reshape(-1, 28*28).astype(float)


def model_predict_prob(X):
    # Reshape the flattened input (N,784) to (N,28,28)
    images = X.reshape(-1, 28, 28)
    return np.array([model.hypothesis(x) for x in images])



num_features = 28 * 28
num_classes = 10
learning_rate = 1e-2

model = MultiLogisticRegressionModel(
    num_features=num_features,
    num_classes=num_classes,
    learning_rate=learning_rate
)
model.train(train_data, evalset=test_data)

background_size = 100
bg_indices = np.random.choice(train_arrays.shape[0], background_size, replace=False)
background = train_arrays[bg_indices]
e = shap.KernelExplainer(model_predict_prob, background)

explained_imgs = []
for i in range(10):
    file_path = os.path.join("mnist_png", "explained_imgs", f"{i}.png")
    img = PILImage.open(file_path).convert('L')
    img_array = np.array(img).astype(float) / 255.0
    explained_imgs.append(img_array)

explained_imgs = np.array(explained_imgs)
explained_imgs_flat = explained_imgs.reshape((10, 28*28))
shap_values = e.shap_values(explained_imgs_flat)
explained_imgs_4d = explained_imgs.reshape((10, 28, 28, 1))

pred_probs = model_predict_prob(explained_imgs_flat)
pred_labels = np.argmax(pred_probs, axis=1)

svals_perimage = []
for i in range(len(pred_labels)):
    p = pred_labels[i]
    svals_perimage.append(shap_values[p][i])

svals_perimage = np.array(svals_perimage)
svals_perimage = svals_perimage.reshape((10, 28, 28, 1))

for i in range(10):
    plt.figure()
    plt.title(f"Image {i} - Predicted class = {pred_labels[i]}")
    plt.imshow(svals_perimage[i, :, :, 0], cmap="RdBu")
    plt.colorbar()
    plt.show()
