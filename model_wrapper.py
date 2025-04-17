import torch
import torch.nn as nn

class PyTorchLogisticWrapper(nn.Module):
    """
    A thin PyTorch wrapper around your trained MultiLogisticRegressionModel.
    This class will let OmniXAI see a recognized 'nn.Module' for gradient-based SHAP.
    """
    def __init__(self, custom_model):
        super().__init__()
        self.custom_model = custom_model

        # Extract your learned weights: shape (num_classes, num_features+1)
        w = self.custom_model.get_weights()  # a NumPy array

        # Convert them to a PyTorch float tensor Parameter
        # shape: (num_classes, num_features+1)
        # We'll store these as a single parameter so PyTorch sees them.
        self.weights = nn.Parameter(
            torch.from_numpy(w).float()
        )

    def forward(self, x):
        """
        x: a PyTorch float tensor of shape (batch_size, num_features).
           For MNIST, num_features=784.

        We do the same logic as your logistic regression:
        1) add a '1' feature for bias
        2) matrix multiply by self.weights
        3) apply softmax
        """
        # shape (batch_size, num_features)
        batch_size = x.shape[0]

        # Add bias feature:
        # We'll create a column of ones and concatenate to x => shape (batch_size, num_features+1).
        ones = torch.ones(batch_size, 1, dtype=x.dtype, device=x.device)
        x_aug = torch.cat([ones, x], dim=1)

        # Weighted logits => (batch_size, num_classes)
        # self.weights is shape (num_classes, num_features+1), so we do x_aug @ self.weights.T
        logits = x_aug @ self.weights.T

        # Return softmax for probabilities
        return torch.softmax(logits, dim=1)