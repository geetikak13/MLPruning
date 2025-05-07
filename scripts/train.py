# --- scripts/train.py ---
import torch
import torch.optim as optim
import torch.nn as nn
import os
import sys
import argparse
# Add src directory to Python path to allow importing modules from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import SimpleCNN # Import the CNN model definition
from src.data_loader import get_cifar10_loaders # Import data loading function
from src.utils import save_model, evaluate_model, get_device # Import utility functions

def train_base_model(epochs, lr, batch_size, data_dir, model_save_path, num_workers):
    """
    Trains the base CNN model on CIFAR-10 using the best available device.

    Args:
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        batch_size (int): Number of samples per batch.
        data_dir (str): Directory to store/load CIFAR-10 data.
        model_save_path (str): Path to save the best performing model checkpoint.
        num_workers (int): Number of worker processes for data loading.
    """
    print("--- Training Base Model ---")
    # Determine the best available device (MPS, CUDA, or CPU)
    device = get_device()

    # Load CIFAR-10 training and test datasets
    trainloader, testloader, _ = get_cifar10_loaders(batch_size, data_dir, num_workers=num_workers)

    # Initialize the CNN model (on CPU initially)
    model = SimpleCNN(num_classes=10)

    # Define the loss function (Cross Entropy for classification)
    criterion = nn.CrossEntropyLoss()
    # Define the optimizer (Adam is a common choice)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Define a learning rate scheduler (decays LR over epochs)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1) # Decrease LR by factor of 0.1 every 15 epochs

    best_accuracy = 0.0 # Track the best test accuracy seen so far
    # Ensure the directory for saving models exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # --- Training Loop ---
    for epoch in range(epochs):
        print(f"\n--- Starting Epoch {epoch+1}/{epochs} ---")
        # Explicitly move model to the target device at the start of each epoch
        # This helps prevent device mismatches, especially observed with MPS
        model.to(device)
        model.train() # Set the model to training mode (enables dropout, etc.)
        # Verify model device (optional debug print)
        # print(f"Ensured model is on device: {next(model.parameters()).device}")

        running_loss = 0.0 # Accumulate loss within the epoch for reporting

        # Iterate over batches of training data
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # Move input data and target labels to the target device
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients accumulated from the previous batch
            optimizer.zero_grad()

            # Forward pass: Get model predictions for the input batch
            try:
                outputs = model(inputs)
            except RuntimeError as e:
                # Catch potential runtime errors (e.g., device mismatches)
                # and provide detailed context for debugging.
                print("\n--- Runtime Error during forward pass ---")
                print(f"Epoch: {epoch+1}, Batch Index: {batch_idx}")
                print(f"Input device: {inputs.device}, Input dtype: {inputs.dtype}")
                try:
                    # Check the device of the model's parameters at the time of error
                    current_param_device = next(model.parameters()).device
                    current_param_dtype = next(model.parameters()).dtype
                    print(f"Model parameter device: {current_param_device}, Model parameter dtype: {current_param_dtype}")
                except StopIteration:
                    print("Model has no parameters.")
                print(f"Error message: {e}")
                print("-----------------------------------------\n")
                raise e # Re-raise the error to halt execution

            # Calculate the loss between predictions and actual targets
            loss = criterion(outputs, targets)

            # Backward pass: Compute gradients of the loss with respect to model parameters
            loss.backward()

            # Update model parameters based on computed gradients
            optimizer.step()

            # Accumulate the loss for reporting average loss
            running_loss += loss.item()

            # Print average loss every 100 batches
            if batch_idx > 0 and batch_idx % 100 == 0:
                print(f'[Epoch {epoch + 1}, Batch {batch_idx + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0 # Reset loss accumulator

        # --- End of Epoch ---
        # Evaluate the model on the test set after each epoch
        print(f"\n--- Evaluating Epoch {epoch+1}/{epochs} ---")
        # evaluate_model handles moving model/data to device internally
        accuracy, avg_loss = evaluate_model(model, testloader, criterion, device)

        # Save the model checkpoint if it achieves the best accuracy so far
        if accuracy > best_accuracy:
            print(f"New best accuracy ({accuracy:.2f}%) > previous best ({best_accuracy:.2f}%). Saving model...")
            # save_model utility moves model to CPU before saving
            save_model(model, model_save_path)
            best_accuracy = accuracy
        else:
            print(f"Accuracy ({accuracy:.2f}%) did not improve from best ({best_accuracy:.2f}%).")

        # Step the learning rate scheduler
        scheduler.step()
        print(f"Current learning rate: {scheduler.get_last_lr()[0]:.6f}")

    # --- End of Training ---
    print(f"\n--- Training Finished ---")
    print(f"Best accuracy achieved: {best_accuracy:.2f}%")
    print(f"Base model saved to: {model_save_path}")

# --- Main Execution Block ---
if __name__ == "__main__":
    # Set up argument parser for command-line configuration
    parser = argparse.ArgumentParser(description='Train a simple CNN on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training and testing.')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for CIFAR-10 data.')
    parser.add_argument('--model_save_path', type=str, default='./models/base_model.pth', help='Path to save the trained base model.')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for DataLoader.')
    # Parse arguments from command line
    args = parser.parse_args()

    # Call the training function with parsed arguments
    train_base_model(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        model_save_path=args.model_save_path,
        num_workers=args.num_workers
    )
