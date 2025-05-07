# --- src/model.py ---
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for CIFAR-10.
    """
    def __init__(self, num_classes=10):
        """
        Initializes the layers of the CNN.

        Args:
            num_classes (int): Number of output classes (default: 10 for CIFAR-10).
        """
        super(SimpleCNN, self).__init__()
        # Convolutional Layer 1: 3 input channels (RGB), 32 output channels, 3x3 kernel, padding 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # Convolutional Layer 2: 32 input channels, 64 output channels, 3x3 kernel, padding 1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Max Pooling Layer 1: 2x2 kernel, stride 2 (downsamples spatial dimensions by 2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Convolutional Layer 3: 64 input channels, 128 output channels, 3x3 kernel, padding 1
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # Convolutional Layer 4: 128 input channels, 128 output channels, 3x3 kernel, padding 1
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # Max Pooling Layer 2: 2x2 kernel, stride 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layer 1 (Linear Layer)
        # Input size calculation: After two 2x2 pooling layers, a 32x32 image becomes 8x8.
        # So, the input features are 128 channels * 8 height * 8 width = 8192.
        # Output features: 512
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        # Fully Connected Layer 2 (Output Layer): 512 input features, num_classes output features
        self.fc2 = nn.Linear(512, num_classes)
        # Dropout layer: Applies dropout with a probability of 0.5 during training for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor (batch_size, 3, 32, 32).

        Returns:
            torch.Tensor: The output tensor (batch_size, num_classes).
        """
        # Block 1: Conv -> ReLU -> Conv -> ReLU -> Pool
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        # Block 2: Conv -> ReLU -> Conv -> ReLU -> Pool
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))

        # Flatten the output from the convolutional layers before feeding into fully connected layers
        # -1 infers the batch size. The remaining dimensions are flattened into a single vector.
        x = x.view(-1, 128 * 8 * 8)

        # Apply dropout before the first fully connected layer
        x = self.dropout(x)
        # First fully connected layer followed by ReLU activation
        x = F.relu(self.fc1(x))
        # Apply dropout before the output layer
        x = self.dropout(x)
        # Output layer (no activation function here, as CrossEntropyLoss applies LogSoftmax internally)
        x = self.fc2(x)
        return x
