import numpy as np
import util
from model import MultiLogisticRegressionModel
import os
from PIL import Image as PILImage
from IPython.display import display

from omnixai.data.image import Image
from omnixai.explainers.vision import L2XImage

from torch import nn


def custom_cross_entropy_loss(input, target):
    target = target.long()
    return nn.CrossEntropyLoss()(input, target)


train_data = util.get_dataset("mnist_train")
test_data = util.get_dataset("mnist_test")

train_arrays = np.array(train_data.xs)
test_arrays = np.array(test_data.xs)

train_labels = np.array(train_data.ys, dtype=np.int64)
test_labels = np.array(test_data.ys, dtype=np.int64)

train_imgs = Image(train_arrays, batched=True)
test_imgs = Image(test_arrays, batched=True)

num_features = 28 * 28
num_classes = 10
learning_rate = 1e-2

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
        probs = model.hypothesis(x_2d)
        preds.append(probs)
    return np.array(preds)

explainer = L2XImage(
    training_data=train_imgs,
    predict_function=multi_logistic_predict_function,
    loss_function=custom_cross_entropy_loss
)

explained_imgs_list = []
for i in range(10):
    file_path = os.path.join("mnist_png", "explained_imgs", f"{i}.png")
    img = PILImage.open(file_path).convert('L')
    img_array = np.array(img).astype(float) / 255.0
    explained_imgs_list.append(img_array)

explained_imgs = Image(np.array(explained_imgs_list), batched=True)
explanations = explainer.explain(explained_imgs)

for idx in range(len(explained_imgs)):
    display(explanations.ipython_plot(index=idx))
predictions = multi_logistic_predict_function(explained_imgs)
print("Predicted probabilities for the custom images:\n", predictions)
