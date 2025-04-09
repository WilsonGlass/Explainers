import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import util
from model import MultiLogisticRegressionModel

from omnixai.data.image import Image
from omnixai.explainers.vision import L2XImage

train_data = util.get_dataset("mnist_train")  # Entire MNIST training set
test_data = util.get_dataset("mnist_test")    # Entire MNIST test set
train_arrays = np.array(train_data.xs)  # shape (N_train, 28, 28)
test_arrays = np.array(test_data.xs)    # shape (N_test, 28, 28)

train_imgs = Image(train_arrays, batched=True)  # Convert to omnixai.data.image.Image
test_imgs = Image(test_arrays, batched=True)

train_labels = np.array(train_data.ys)
test_labels = np.array(test_data.ys)

num_features = 28 * 28
num_classes = 10
learning_rate = 1e-2

model = MultiLogisticRegressionModel(
    num_features=num_features,
    num_classes=num_classes,
    learning_rate=learning_rate
)
model.train(train_data, evalset=test_data)  # You can remove evalset if desired

def multi_logistic_predict_function(images):
    """
    images: a list (or array) of omnixai.data.image.Image
    returns: a numpy array of shape [N, 10], each row is a softmax distribution
    """
    preds = []
    for im in images:
        # Convert to numpy, shape => (28, 28)
        x_2d = im.to_numpy().astype(float)
        # model.hypothesis(x_2d) => returns length-10 distribution
        probs = model.hypothesis(x_2d)
        preds.append(probs)
    return np.array(preds)  # shape => (N, 10)

explainer = L2XImage(
    training_data=train_imgs,
    predict_function=multi_logistic_predict_function
)

# Explain the first 5 test images
explanations = explainer.explain(test_imgs[0:5])

# Plot one explanation by index
explanations.ipython_plot(index=1)

predictions = multi_logistic_predict_function(test_imgs[0:10])
print("Predicted probabilities for the first 10 test images:\n", predictions)
