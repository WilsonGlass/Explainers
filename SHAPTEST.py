import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from PIL import Image as PILImage
from IPython.display import display

# OmniXAI imports
from omnixai.data.image import Image
from omnixai.explainers.vision import ShapImage


# ------------------------------------------------------------------
# 1) Define a standard PyTorch logistic regression for MNIST
# ------------------------------------------------------------------
class SimpleLogistic(nn.Module):
    def __init__(self, in_features=784, out_classes=10):
        super().__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        # x shape: (batch_size, 784)
        logits = self.linear(x)
        return torch.softmax(logits, dim=1)  # shape (batch_size, 10)


# ------------------------------------------------------------------
# 2) Create a small PyTorch Dataset for your MNIST data
# ------------------------------------------------------------------
class NumpyMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        # images: shape (N, 28, 28) or (N, 784)
        # labels: shape (N,)
        self.images = images.reshape(-1, 784).astype(np.float32) / 255.0
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]
        return x, y


# ------------------------------------------------------------------
# 3) Train the logistic model on MNIST with standard PyTorch
# ------------------------------------------------------------------
def train_pytorch_logistic(train_images, train_labels, test_images, test_labels, epochs=5, batch_size=64):
    from torch.utils.data import DataLoader

    # 3a. Create Datasets and DataLoaders
    train_dataset = NumpyMNISTDataset(train_images, train_labels)
    test_dataset  = NumpyMNISTDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # 3b. Create model, loss, optimizer
    model = SimpleLogistic(in_features=784, out_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    # 3c. Train loop
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            # X_batch shape: (batch_size, 784)
            # y_batch shape: (batch_size,)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

        # optional: compute accuracy on test set
        test_acc = evaluate_accuracy(model, test_loader)
        print(f"Epoch {epoch+1}/{epochs}, Test accuracy: {test_acc*100:.2f}%")

    return model


def evaluate_accuracy(model, test_loader):
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch)              # shape (batch_size, 10)
            pred_labels = preds.argmax(dim=1)   # shape (batch_size,)
            correct += (pred_labels == y_batch).sum().item()
            total   += len(y_batch)
    return correct / total


# ------------------------------------------------------------------
# 4) OmniXAI preprocess function for logistic regression
# ------------------------------------------------------------------
def logistic_preprocess(img: Image):
    """
    This function is mandatory because your OmniXAI version requires
    `preprocess_function` as a positional argument to ShapImage.

    1) Convert the Image object to a NumPy float array, squeeze out extra dims.
    2) If it's a single image => shape (28,28)
       If it's multiple images => shape (B, 28, 28)
    3) Flatten to (batch_size, 784)
    4) Convert to a PyTorch float tensor, normalized by 255 if not already.
    """
    arr = img.to_numpy().astype(np.float32)  # might be shape (28,28) or (B,28,28)
    arr = np.squeeze(arr)

    # Single image => (28,28)
    if arr.ndim == 2:
        arr = arr.reshape(1, 784)  # (1,784)
    # Batch => (B,28,28)
    elif arr.ndim == 3:
        B = arr.shape[0]
        arr = arr.reshape(B, 784)
    else:
        raise ValueError(f"Unexpected shape {arr.shape} in logistic_preprocess")

    # If needed, scale from [0,255] => [0,1], but if you've already done so earlier, skip
    # arr /= 255.0

    return torch.from_numpy(arr)  # shape (batch_size,784)


# ------------------------------------------------------------------
# 5) Example main code using the approach
# ------------------------------------------------------------------
if __name__ == "__main__":
    import util
    import numpy as np

    # Load MNIST from your utility
    train_data = util.get_dataset("mnist_train")
    test_data  = util.get_dataset("mnist_test")

    train_arrays = np.array(train_data.xs)   # shape (N_train, 28, 28)
    train_labels = np.array(train_data.ys)   # shape (N_train,)
    test_arrays  = np.array(test_data.xs)    # shape (N_test, 28, 28)
    test_labels  = np.array(test_data.ys)    # shape (N_test,)

    # Train the PyTorch logistic model
    model = train_pytorch_logistic(
        train_images=train_arrays,
        train_labels=train_labels,
        test_images=test_arrays,
        test_labels=test_labels,
        epochs=5,        # adjust as needed
        batch_size=64    # adjust as needed
    )

    # Create some sample images for explanation
    explained_imgs_list = []
    for i in range(10):
        file_path = os.path.join("mnist_png", "explained_imgs", f"{i}.png")
        img = PILImage.open(file_path).convert('L')
        img_array = np.array(img).astype(np.float32)
        explained_imgs_list.append(img_array)

    explained_imgs = Image(np.array(explained_imgs_list), batched=True)

    # Build your ShapImage explainer
    # Because your OmniXAI version demands a preprocess_function, pass logistic_preprocess
    training_data = Image(test_arrays[:100], batched=True)  # reference set for SHAP
    explainer = ShapImage(
        training_data=training_data,
        preprocess_function=logistic_preprocess,
        model=model,
        mode="classification",
        shap_values_params={"check_additivity": False}
    )

    # Explain them
    explanations = explainer.explain(explained_imgs)
    for idx in range(len(explained_imgs)):
        display(explanations.ipython_plot(index=idx))

    # (Optional) Print predictions
    with torch.no_grad():
        model.eval()
        for i, arr in enumerate(explained_imgs_list):
            x = torch.from_numpy(arr.reshape(1, 784).astype(np.float32))
            probs = model(x)[0].numpy()
            print(f"Image {i}, predicted probs = {probs}, label = {probs.argmax()}")
