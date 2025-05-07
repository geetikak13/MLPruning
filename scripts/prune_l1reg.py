# --- scripts/prune_l1reg.py ---
import torch
import torch.optim as optim
import torch.nn as nn
import os
import sys
import argparse
import copy # To potentially restore best model if fine-tuning hurts
# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import SimpleCNN
from src.data_loader import get_cifar10_loaders
from src.utils import save_model, load_model, evaluate_model, calculate_sparsity, get_device

def apply_l1_regularization(model, loss, lambda_l1):
    """
    Calculates the L1 penalty for model weights and adds it to the loss.
    Uses .abs().sum() for compatibility with multi-dimensional tensors.

    Args:
        model (torch.nn.Module): The model whose weights are penalized.
        loss (torch.Tensor): The original calculated loss (e.g., cross-entropy).
        lambda_l1 (float): The L1 regularization strength coefficient.

    Returns:
        torch.Tensor: The total loss (original loss + L1 penalty).
    """
    l1_penalty = torch.tensor(0., device=loss.device) # Initialize penalty on same device as loss
    # Iterate through all model parameters
    for param in model.parameters():
        # Apply penalty only to parameters that require gradients and are weights (typically > 1 dimension)
        if param.requires_grad and param.dim() > 1:
            # Use sum of absolute values for multi-dimensional tensors
            l1_penalty += torch.abs(param).sum()
    # Add the scaled penalty to the original loss
    return loss + lambda_l1 * l1_penalty

def threshold_model(model, threshold):
    """
    Applies hard thresholding to model weights in-place.
    Sets weights with absolute value below the threshold to zero.

    Args:
        model (torch.nn.Module): The model to threshold.
        threshold (float): The magnitude threshold.
    """
    print(f"\nApplying weight threshold: {threshold:.1E}")
    # Ensure operations are done without tracking gradients
    with torch.no_grad():
        # Iterate through all model parameters
        for param in model.parameters():
            # Apply only to weights (typically > 1 dimension) that require gradients
            if param.requires_grad and param.dim() > 1:
                # Use torch.where to set values below threshold to 0, keep others unchanged
                param.data = torch.where(torch.abs(param.data) < threshold, torch.zeros_like(param.data), param.data)
    print("Thresholding complete.")
    # Report sparsity immediately after thresholding
    calculate_sparsity(model)

def train_l1_regularized(model, trainloader, testloader, criterion, optimizer, scheduler, device, epochs, lambda_l1, model_save_path_l1):
    """
    Trains a model using a specified L1 regularization strength. Saves the best performing model during L1 training.

    Args:
        model: The model to train.
        trainloader: DataLoader for training data.
        testloader: DataLoader for test data.
        criterion: The base loss function.
        optimizer: The optimizer for training.
        scheduler: Learning rate scheduler.
        device: The target device.
        epochs (int): Number of epochs for L1 training.
        lambda_l1 (float): L1 regularization strength.
        model_save_path_l1 (str): Path to save the best model during this phase.

    Returns:
        float: The best test accuracy achieved during L1 training.
    """
    print(f"\n--- Training with L1 Regularization (Lambda={lambda_l1:.1E}) ---")
    best_l1_accuracy = 0.0
    os.makedirs(os.path.dirname(model_save_path_l1), exist_ok=True)

    for epoch in range(epochs):
        print(f"\n--- L1 Training Epoch {epoch+1}/{epochs} ---")
        model.to(device); model.train() # Move to device and set train mode
        # print(f"Ensured model is on device: {next(model.parameters()).device}")
        running_loss = 0.0; running_l1_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device); optimizer.zero_grad()
            try: outputs = model(inputs)
            except RuntimeError as e: print(f"RuntimeError: {e}"); raise e

            base_loss = criterion(outputs, targets) # Calculate base loss
            total_loss = apply_l1_regularization(model, base_loss, lambda_l1) # Add L1 penalty

            total_loss.backward(); optimizer.step() # Backpropagate and update weights

            running_loss += base_loss.item(); running_l1_loss += total_loss.item()
            if batch_idx > 0 and batch_idx % 100 == 0:
                print(f'[Epoch {epoch + 1}, Batch {batch_idx + 1:5d}] Base Loss: {running_loss / 100:.3f} | L1 Loss: {running_l1_loss / 100:.3f}')
                running_loss = 0.0; running_l1_loss = 0.0

        # Evaluate after each epoch
        print(f"\n--- Evaluating Epoch {epoch+1}/{epochs} ---")
        accuracy, avg_loss = evaluate_model(model, testloader, criterion, device)
        calculate_sparsity(model) # Check sparsity during L1 training

        # Save model if it's the best so far in this phase
        if accuracy > best_l1_accuracy:
            print(f"New best L1 accuracy ({accuracy:.2f}%). Saving L1 model...")
            save_model(model, model_save_path_l1); best_l1_accuracy = accuracy

        scheduler.step(); print(f"Current learning rate: {scheduler.get_last_lr()[0]:.6f}")

    print(f"\n--- L1 Training Finished. Best L1 Accuracy: {best_l1_accuracy:.2f}% ---")
    print(f"Best L1 regularized model during training saved to: {model_save_path_l1}")
    return best_l1_accuracy


def fine_tune_thresholded(model, trainloader, testloader, criterion, optimizer, scheduler, device, epochs, model_save_path_final):
    """
    Fine-tunes a thresholded model (typically without L1 penalty). Saves the best performing model during fine-tuning.

    Args:
        model: The thresholded model to fine-tune.
        trainloader: DataLoader for training data.
        testloader: DataLoader for test data.
        criterion: The loss function.
        optimizer: The optimizer for fine-tuning.
        scheduler: Learning rate scheduler for fine-tuning.
        device: The target device.
        epochs (int): Number of fine-tuning epochs.
        model_save_path_final (str): Path to save the best fine-tuned model.

    Returns:
        str: Path to the best model saved during fine-tuning (could be pre-FT state).
    """
    print("\n--- Fine-tuning Thresholded Model ---")
    best_ft_accuracy = 0.0
    os.makedirs(os.path.dirname(model_save_path_final), exist_ok=True)

    # Evaluate the model's performance right after thresholding, before fine-tuning
    print("Evaluating thresholded model before fine-tuning...")
    pre_ft_accuracy, _ = evaluate_model(model, testloader, criterion, device)
    best_ft_accuracy = pre_ft_accuracy # Initialize best accuracy with the post-threshold accuracy

    # Save the initially thresholded model state in case fine-tuning makes it worse
    thresholded_save_path = model_save_path_final.replace('.pth', '_thresholded_only.pth')
    print(f"Saving thresholded state to {thresholded_save_path}...")
    save_model(model, thresholded_save_path)
    # Assume the best model initially is the thresholded one
    best_model_state_path = thresholded_save_path


    for epoch in range(epochs):
        print(f"\n--- Fine-tuning Epoch {epoch+1}/{epochs} ---")
        model.to(device); model.train() # Move to device and set train mode
        # print(f"Ensured model is on device: {next(model.parameters()).device}")
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device); optimizer.zero_grad()
            try: outputs = model(inputs)
            except RuntimeError as e: print(f"RuntimeError: {e}"); raise e

            loss = criterion(outputs, targets) # Calculate loss (NO L1 penalty here)

            loss.backward(); optimizer.step() # Backpropagate and update
            running_loss += loss.item()
            if batch_idx > 0 and batch_idx % 100 == 0:
                print(f'[Epoch {epoch + 1}, Batch {batch_idx + 1:5d}] Fine-tune Loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        # Evaluate after each fine-tuning epoch
        print(f"\n--- Evaluating Epoch {epoch+1}/{epochs} ---")
        accuracy, avg_loss = evaluate_model(model, testloader, criterion, device)

        # Save model if it's the best fine-tuned version so far
        if accuracy >= best_ft_accuracy: # Use >= to prefer later epochs if accuracy is same
            print(f"New best fine-tune accuracy ({accuracy:.2f}%). Saving final model...")
            save_model(model, model_save_path_final); best_ft_accuracy = accuracy
            best_model_state_path = model_save_path_final # Update best path
        else:
             print(f"Accuracy ({accuracy:.2f}%) did not improve from best ({best_ft_accuracy:.2f}%).")


        scheduler.step(); print(f"Current learning rate: {scheduler.get_last_lr()[0]:.6f}")

    print(f"\n--- Fine-tuning Finished. Best Fine-tune Accuracy: {best_ft_accuracy:.2f}% ---")
    # If the best accuracy was the pre-fine-tuning one, the final path might point to the _thresholded_only file
    print(f"Final L1 pruned model state saved to: {best_model_state_path}")
    # Return the path to the actually saved final file (which might be the _thresholded_only one)
    return best_model_state_path


def main(args):
    """
    Main function to orchestrate L1 regularization training, thresholding, and fine-tuning.
    """
    print("--- Pruning Script (L1 Regularization Mode) ---"); device = get_device()
    trainloader, testloader, _ = get_cifar10_loaders(args.batch_size, args.data_dir, num_workers=args.num_workers)
    model = SimpleCNN(num_classes=10) # Init on CPU

    # Load base model if specified, otherwise train from scratch with L1
    if args.base_model_path and os.path.exists(args.base_model_path):
        print(f"Loading base model from: {args.base_model_path} to fine-tune with L1...")
        if not load_model(model, args.base_model_path, device): print("Exiting: Failed to load base model."); return
        # If starting from base, use the main LR for the L1 phase
        l1_lr = args.lr
        l1_epochs = args.l1_epochs
    else:
        print("No valid base model path provided or found. Training from scratch with L1.")
        model.to(device) # Move model if training from scratch
        l1_lr = args.lr
        l1_epochs = args.l1_epochs

    criterion = nn.CrossEntropyLoss()
    # Use separate optimizers/schedulers for L1 training and fine-tuning
    optimizer_l1 = optim.Adam(model.parameters(), lr=l1_lr)
    # Adjust scheduler step size based on number of L1 epochs
    scheduler_l1 = torch.optim.lr_scheduler.StepLR(optimizer_l1, step_size=max(1, l1_epochs // 3), gamma=0.1)

    # Define save paths
    l1_model_dir = os.path.dirname(args.final_model_save_path)
    # Temporary path to save the best model during the L1 training phase
    model_save_path_l1_temp = os.path.join(l1_model_dir, f"l1_trained_lambda_{args.l1_lambda:.1E}_temp.pth")
    os.makedirs(l1_model_dir, exist_ok=True)

    # 1. Train with L1 Regularization
    train_l1_regularized(model, trainloader, testloader, criterion, optimizer_l1, scheduler_l1, device, l1_epochs, args.l1_lambda, model_save_path_l1_temp)

    # Load the best L1 model achieved during that phase for thresholding
    print(f"\nLoading best L1 trained model from {model_save_path_l1_temp} for thresholding...")
    # Create a new model instance to load into, ensuring clean state
    model_for_thresholding = SimpleCNN(num_classes=10)
    if not load_model(model_for_thresholding, model_save_path_l1_temp, device): print("Exiting: Failed to load L1 model."); return
    model = model_for_thresholding # Use the loaded model

    # 2. Apply Thresholding (in-place)
    threshold_model(model, args.threshold)

    # 3. Fine-tune the thresholded model (optional)
    if args.fine_tune_epochs > 0:
        # Use a potentially different (usually lower) LR for fine-tuning
        optimizer_ft = optim.Adam(model.parameters(), lr=args.fine_tune_lr)
        # Use a potentially different scheduler for fine-tuning
        scheduler_ft = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=max(1, args.fine_tune_epochs // 3), gamma=0.2) # Slower decay?
        fine_tune_thresholded(model, trainloader, testloader, criterion, optimizer_ft, scheduler_ft, device, args.fine_tune_epochs, args.final_model_save_path)
    else:
        print("\nSkipping fine-tuning step.")
        # Save the thresholded model directly if no fine-tuning is requested
        print(f"Saving thresholded (non-fine-tuned) model to {args.final_model_save_path}...")
        save_model(model, args.final_model_save_path)

    # Clean up temporary L1 model file
    if os.path.exists(model_save_path_l1_temp):
        try:
            os.remove(model_save_path_l1_temp)
            print(f"Removed temporary file: {model_save_path_l1_temp}")
        except OSError as e:
            print(f"Error removing temporary file {model_save_path_l1_temp}: {e}")


    print("\n--- L1 Pruning Script Finished ---")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='L1 Regularization Pruning Script for CIFAR-10 CNN')
    parser.add_argument('--base_model_path', type=str, default=None,
                        help='Optional: Path to pre-trained base model to start L1 fine-tuning from.')
    parser.add_argument('--final_model_save_path', type=str, default='./models/pruned_l1reg/final_l1_pruned_model.pth',
                        help='Path to save the final pruned model after thresholding and optional fine-tuning.')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory for CIFAR-10 data.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for data loaders.')
    parser.add_argument('--l1_epochs', type=int, default=50,
                        help='Number of epochs to train/fine-tune with L1 regularization.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for L1 training phase (or initial LR if starting from scratch).')
    parser.add_argument('--l1_lambda', type=float, required=True,
                        help='L1 regularization strength (e.g., 1e-4, 1e-5).')
    parser.add_argument('--threshold', type=float, required=True,
                        help='Magnitude threshold below which weights are set to zero after L1 training (e.g., 1e-3, 1e-4).')
    parser.add_argument('--fine_tune_epochs', type=int, default=10,
                        help='Number of epochs to fine-tune after thresholding (set to 0 to skip).')
    parser.add_argument('--fine_tune_lr', type=float, default=0.0001,
                        help='Learning rate for the post-thresholding fine-tuning phase.')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers for DataLoader.')
    args = parser.parse_args(); main(args)
