# --- scripts/prune_igreedy.py ---
import torch
import torch.optim as optim
import torch.nn as nn
import os
import sys
import argparse
# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import SimpleCNN
from src.data_loader import get_cifar10_loaders
from src.utils import load_model, save_model, evaluate_model, calculate_sparsity, get_device
# Import the specific pruning function for this method
from src.pruning import iterative_pruning_finetuning

def main(args):
    """
    Loads a base model and performs iterative greedy magnitude pruning
    until a specified accuracy threshold is met.
    """
    print("--- Pruning Script (iGreedy Threshold Mode) ---")
    # Determine the execution device
    device = get_device()

    # Load datasets
    trainloader, testloader, _ = get_cifar10_loaders(args.batch_size, args.data_dir, num_workers=args.num_workers)

    # Initialize model architecture (on CPU initially)
    model = SimpleCNN(num_classes=10)

    # Load the pre-trained base model state onto the target device
    print(f"Loading base model from: {args.base_model_path}")
    if not load_model(model, args.base_model_path, device):
        print("Exiting: Failed to load base model.")
        return # Exit if base model cannot be loaded

    # Define loss function and optimizer class (instance created within pruning function)
    criterion = nn.CrossEntropyLoss()
    optimizer_class = optim.Adam

    # Ensure the output directory for pruned models exists
    os.makedirs(args.pruned_model_dir, exist_ok=True)

    # Call the iterative pruning function which handles the threshold logic
    best_model_path = iterative_pruning_finetuning(
        model=model,                            # The model instance (already on device)
        trainloader=trainloader,                # Training data loader
        testloader=testloader,                  # Testing data loader
        criterion=criterion,                    # Loss function
        optimizer_class=optimizer_class,        # Optimizer type
        base_lr=args.fine_tune_lr,              # Initial LR for fine-tuning
        prune_step_amount=args.prune_step_amount, # Amount to prune each step
        accuracy_threshold=args.accuracy_threshold, # Target accuracy floor
        fine_tune_epochs=args.fine_tune_epochs, # Epochs for fine-tuning
        device=device,                          # Target device
        model_save_dir=args.pruned_model_dir    # Directory to save models meeting threshold
    )

    # Report the outcome
    if best_model_path:
         print(f"\nPruning finished. Best model meeting threshold saved at: {best_model_path}")
    else:
         print("\nPruning finished. No model met the specified accuracy threshold (or initial accuracy was too low).")

    print("--- iGreedy Pruning Script Finished ---")

# --- Main Execution Block ---
if __name__ == "__main__":
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description='Iterative Greedy Pruning Script for CIFAR-10 CNN (Threshold Mode)')
    parser.add_argument('--base_model_path', type=str, default='./models/base_model.pth',
                        help='Path to the pre-trained base model file.')
    parser.add_argument('--pruned_model_dir', type=str, default='./models/pruned_threshold',
                        help='Directory to save pruned model checkpoints meeting the threshold.')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory for CIFAR-10 data.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for data loaders.')
    parser.add_argument('--prune_step_amount', type=float, default=0.2,
                        help='Fraction of remaining weights to prune per step (e.g., 0.1 for 10%).')
    parser.add_argument('--accuracy_threshold', type=float, required=True,
                        help='Minimum test accuracy percentage required to continue pruning (e.g., 80.0).')
    parser.add_argument('--fine_tune_epochs', type=int, default=5,
                        help='Number of epochs to fine-tune after each pruning step.')
    parser.add_argument('--fine_tune_lr', type=float, default=0.0001,
                        help='Base learning rate for fine-tuning (will decay).')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers for DataLoader.')
    # Parse arguments
    args = parser.parse_args()
    # Run the main function
    main(args)
