import numpy as np
np.seterr(over='ignore', invalid='ignore', divide='ignore')
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import math
import util


class Model:
    """
    Abstract class for the machine learning models.
    """

    def get_features(self, x_input):
        pass

    def get_weights(self):
        pass

    def hypothesis(self, x):
        pass

    def predict(self, x):
        pass

    def loss(self, x, y):
        pass

    def gradient(self, x, y):
        pass

    def train(self, dataset):
        pass


class MultiLogisticRegressionModel(Model):
    """
    Multinomial logistic regression model with image-pixel features (num_features = image size, e.g., 28x28 = 784 for MNIST).
    x is a 2-D image, represented as a list of lists (28x28 for MNIST). y is an integer between 1 and num_classes.
    The goal is to fit P(y = k | x) = hypothesis(x)[k], where hypothesis is a discrete distribution (list of probabilities)
    over the K classes, and to make a class prediction using the hypothesis.
    """

    def __init__(self, num_features, num_classes, learning_rate=1e-2):
        self.lr = learning_rate
        self.num_classes = num_classes
        self.num_features = num_features
        self.weights = np.zeros((num_classes, num_features + 1))
        self.max_iters = 60000
        self.train_accuracies = []
        self.eval_accuracies = []
        self.eval_iters = []

    def get_features(self, x):
        flat = np.array(x).flatten()
        return np.concatenate(([1.0], flat), axis=0)

    def get_weights(self):
        return self.weights

    def hypothesis(self, x):
        f = self.get_features(x)
        logits = [np.dot(self.weights[k], f) for k in range(self.num_classes)]
        exps = np.exp(logits)
        sum_exps = np.sum(exps)
        return exps / sum_exps  # softmax

    def predict(self, x):
        probs = self.hypothesis(x)
        return int(np.argmax(probs))

    def loss(self, x, y):
        probs = self.hypothesis(x)
        eps = 1e-15
        return -np.log(probs[y] + eps)

    def gradient(self, x, y):
        f = self.get_features(x)
        probs = self.hypothesis(x)
        grad = np.zeros_like(self.weights)
        for k in range(self.num_classes):
            indicator = 1.0 if (k == y) else 0.0
            grad[k] = (probs[k] - indicator) * f
        return grad

    def train(self, dataset, evalset = None):
        self.train_accuracies = []
        self.eval_accuracies = []
        self.eval_iters = []

        for i in range(self.max_iters):
            x_samp, y_samp = dataset.get_sample()
            grad = self.gradient(x_samp, y_samp)
            self.weights -= self.lr * grad

            if (i + 1) % 100 == 0:
                train_acc = dataset.compute_average_accuracy(self, step=10)
                self.train_accuracies.append(train_acc)
                if evalset is not None:
                    test_acc = evalset.compute_average_accuracy(self, step=10)
                    self.eval_accuracies.append(test_acc)
                else:
                    self.eval_accuracies.append(None)
                self.eval_iters.append(i + 1)


# PA4 Q6
def multi_classification():
    train_data = util.get_dataset("mnist_train")
    test_data = util.get_dataset("mnist_test")

    model = MultiLogisticRegressionModel(
        num_features=28 * 28,
        num_classes=10,
        learning_rate=1e-2
    )
    model.train(train_data, evalset=test_data)

    print("[Q6] Plotting training accuracy curve:")
    train_data.plot_accuracy_curve(
        model.eval_iters,
        model.train_accuracies,
        title="Training Accuracy (Multi-class)"
    )

    print("[Q6] Plotting confusion matrix on the test set:")
    test_data.plot_confusion_matrix(model, step=10, show_diagonal=False)

    # Plot learned weights for each digit (0..9)
    print("[Q6] Visualizing each digit's learned weights (no bias).")
    for k in range(model.num_classes):
        w_no_bias = model.weights[k][1:]
        w_image = w_no_bias.reshape((28, 28))
        plt.figure()
        plt.imshow(w_image)
        plt.colorbar()
        plt.title(f"Weights for digit {k}")
        plt.show()


def main():
    multi_classification()

if __name__ == "__main__":
    main()
