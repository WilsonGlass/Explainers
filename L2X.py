import numpy as np
import util
from model import MultiLogisticRegressionModel

from omnixai.data.image import Image
from omnixai.explainers.vision import L2XImage

import torch
import torch.nn as nn

# Define a custom loss function that wraps CrossEntropyLoss and casts targets to torch.long.
def custom_cross_entropy_loss(input, target):
    # Cast the target tensor to the correct type (int64 / torch.long)
    target = target.long()
    return nn.CrossEntropyLoss()(input, target)

# Load MNIST data
train_data = util.get_dataset("mnist_train")  # Entire MNIST training set
test_data = util.get_dataset("mnist_test")    # Entire MNIST test set

train_arrays = np.array(train_data.xs)  # shape: (N_train, 28, 28)
test_arrays = np.array(test_data.xs)      # shape: (N_test, 28, 28)

# Prepare true labels explicitly as int64 (though these won't be used by L2XImage)
train_labels = np.array(train_data.ys, dtype=np.int64)
test_labels = np.array(test_data.ys, dtype=np.int64)

# Create Image objects without passing labels (the Image constructor does not accept a labels argument)
train_imgs = Image(train_arrays, batched=True)
test_imgs = Image(test_arrays, batched=True)

num_features = 28 * 28
num_classes = 10
learning_rate = 1e-2

# Initialize and train your MultiLogisticRegressionModel
model = MultiLogisticRegressionModel(
    num_features=num_features,
    num_classes=num_classes,
    learning_rate=learning_rate
)
model.train(train_data, evalset=test_data)

def multi_logistic_predict_function(images):
    """
    images: a list (or array) of omnixai.data.image.Image
    returns: a numpy array of shape [N, 10], each row is a softmax probability distribution
    """
    preds = []
    for im in images:
        # Convert image to numpy array with shape (28, 28)
        x_2d = im.to_numpy().astype(float)
        # Obtain the probability distribution from your model
        probs = model.hypothesis(x_2d)
        preds.append(probs)
    return np.array(preds)  # shape: (N, 10)

# Create the L2X explainer, passing the custom loss function.
explainer = L2XImage(
    training_data=train_imgs,
    predict_function=multi_logistic_predict_function,
    loss_function=custom_cross_entropy_loss
)

# Explain the first 5 test images and plot one explanation by index.
explanations = explainer.explain(test_imgs[0:5])
explanations.ipython_plot(index=1)

# Compute and print predictions for the first 10 test images.
predictions = multi_logistic_predict_function(test_imgs[0:10])
print("Predicted probabilities for the first 10 test images:\n", predictions)
