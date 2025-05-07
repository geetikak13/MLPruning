# --- src/pruning.py ---
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import os
import copy
from .utils import calculate_sparsity, evaluate_model, save_model, load_model
from .model import SimpleCNN

# Renamed function to reflect its action: pruning to a target global sparsity
def prune_model_globally_to_sparsity(model, target_total_sparsity_fraction):
    """
    Applies global unstructured magnitude pruning to achieve a specific
    target overall sparsity level across Conv2D and Linear layer weights.
    Makes the pruning permanent by removing reparameterization buffers.

    Args:
        model (torch.nn.Module): The model to prune (should be on target device).
        target_total_sparsity_fraction (float): The desired overall sparsity
                                                 level (0.0 to 1.0).

    Returns:
        bool: True if pruning was applied, False otherwise.
    """
    parameters_to_prune = []
    # Identify prunable parameters
    for module_name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight') and module.weight.requires_grad:
                parameters_to_prune.append((module, 'weight'))

    if not parameters_to_prune:
        print("No prunable parameters found.")
        return False

    # Clamp target sparsity to valid range [0, 1]
    target_total_sparsity_fraction = max(0.0, min(1.0, target_total_sparsity_fraction))

    print(f"Attempting to prune model globally to {target_total_sparsity_fraction*100:.2f}% sparsity...")

    try:
        # Apply global unstructured pruning directly using the target sparsity level
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=target_total_sparsity_fraction, # Target overall sparsity
        )
    except Exception as e:
        print(f"Error during pruning: {e}")
        # This might happen if target sparsity is less than current, though handled below.
        return False

    # Make pruning permanent
    for module, param_name in parameters_to_prune:
        if prune.is_pruned(module):
             try:
                 prune.remove(module, param_name)
             except Exception as e:
                 print(f"Warning: Could not remove pruning buffers for {module_name}: {e}")
    print("Pruning made permanent (buffers removed).")
    return True

# Updated iterative function
def iterative_pruning_finetuning(model, trainloader, testloader, criterion, optimizer_class, base_lr,
                      prune_step_amount, accuracy_threshold, fine_tune_epochs, device,
                      model_save_dir):
    """
    Performs iterative greedy magnitude pruning and fine-tuning until accuracy
    drops below a threshold. Prunes a fraction of remaining weights each step
    by calculating a target global sparsity level.

    Args:
        model (torch.nn.Module): The model to prune (should be loaded on `device`).
        trainloader: DataLoader for training data.
        testloader: DataLoader for test data.
        criterion: Loss function.
        optimizer_class: The class of the optimizer (e.g., torch.optim.Adam).
        base_lr (float): Base learning rate for fine-tuning.
        prune_step_amount (float): Fraction of *remaining* non-zero weights to target
                                   for removal each step (e.g., 0.1 for 10%).
        accuracy_threshold (float): The minimum acceptable accuracy percentage (e.g., 80.0).
        fine_tune_epochs (int): Number of epochs to fine-tune after each prune.
        device (torch.device): The torch.device to run on ('mps', 'cuda', or 'cpu').
        model_save_dir (str): Directory to save intermediate pruned models meeting the threshold.

    Returns:
        str: Path to the last saved model that met the accuracy threshold, or None.
    """
    print("\n--- Starting Iterative Pruning & Fine-tuning (Threshold Mode - Target Sparsity) ---")
    print(f"Target Accuracy Threshold: {accuracy_threshold:.2f}%")
    print(f"Pruning Step Amount (of remaining): {prune_step_amount*100:.1f}%")

    model.to(device)
    current_accuracy, _ = evaluate_model(model, testloader, criterion, device)
    current_sparsity_percent, _, total_weights = calculate_sparsity(model)
    print(f"Initial Accuracy: {current_accuracy:.2f}%")
    print(f"Initial Sparsity: {current_sparsity_percent:.2f}%")

    if current_accuracy < accuracy_threshold:
        print("Initial model accuracy is already below the threshold. No pruning performed.")
        return None

    os.makedirs(model_save_dir, exist_ok=True)
    last_good_model_path = None
    iteration = 0

    # Save the initial state
    initial_save_path = os.path.join(model_save_dir, f'pruned_model_iter_{iteration}_sparsity_{current_sparsity_percent:.1f}.pth')
    save_model(model, initial_save_path)
    last_good_model_path = initial_save_path
    print(f"Saved initial model state to: {initial_save_path}")

    # --- Iteration Loop ---
    while True:
        iteration += 1
        print(f"\n--- Pruning Iteration {iteration} ---")

        # --- Calculate Target Sparsity ---
        current_sparsity_fraction = current_sparsity_percent / 100.0
        remaining_fraction = 1.0 - current_sparsity_fraction
        # If remaining fraction is effectively zero, stop
        if remaining_fraction <= 1e-9:
             print("Model is fully sparse. Stopping.")
             break

        # Fraction of *total* weights to remove in this step
        fraction_of_total_to_remove = remaining_fraction * prune_step_amount
        # New target overall sparsity fraction
        target_total_sparsity_fraction = current_sparsity_fraction + fraction_of_total_to_remove
        # Clamp to prevent exceeding 1.0 due to floating point issues
        target_total_sparsity_fraction = min(target_total_sparsity_fraction, 1.0)

        # --- Pruning Step ---
        model.to(device) # Ensure model is on device before pruning
        pruning_done = prune_model_globally_to_sparsity(model, target_total_sparsity_fraction)

        if not pruning_done:
            print("Pruning step did not apply (target sparsity likely not increased). Stopping.")
            break # Exit the loop if no pruning happened

        # Recalculate sparsity after pruning
        current_sparsity_percent, _, _ = calculate_sparsity(model)

        # --- Fine-tuning Step ---
        if fine_tune_epochs > 0:
            print(f"Fine-tuning for {fine_tune_epochs} epochs on {device}...")
            current_lr = base_lr / (2**(iteration // 3))
            optimizer = optimizer_class(model.parameters(), lr=current_lr)
            print(f"Using fine-tuning LR: {current_lr:.6f}")
            for epoch in range(fine_tune_epochs):
                model.to(device); model.train()
                # print(f"  Fine-tune Epoch {epoch+1}/{fine_tune_epochs} - Ensuring model is on: {next(model.parameters()).device}")
                running_loss = 0.0
                for batch_idx, (inputs, targets) in enumerate(trainloader):
                    inputs, targets = inputs.to(device), targets.to(device); optimizer.zero_grad()
                    try: outputs = model(inputs)
                    except RuntimeError as e:
                        print("\n--- Runtime Error during fine-tuning forward pass ---"); print(f"Pruning Iter: {iteration}, Fine-tune Epoch: {epoch+1}, Batch Index: {batch_idx}")
                        print(f"Input device: {inputs.device}, Input dtype: {inputs.dtype}")
                        try: current_param_device = next(model.parameters()).device; current_param_dtype = next(model.parameters()).dtype; print(f"Model parameter device: {current_param_device}, Model parameter dtype: {current_param_dtype}")
                        except StopIteration: print("Model has no parameters.")
                        print(f"Error message: {e}"); print("---------------------------------------------------\n"); raise e
                    loss = criterion(outputs, targets); loss.backward(); optimizer.step(); running_loss += loss.item()
                avg_epoch_loss = running_loss / len(trainloader) if len(trainloader) > 0 else 0
                print(f"  Fine-tune Epoch {epoch+1}/{fine_tune_epochs} - Avg Loss: {avg_epoch_loss:.4f}")
        else: print("Skipping fine-tuning.")

        # --- Evaluation Step ---
        model.to(device)
        current_accuracy, _ = evaluate_model(model, testloader, criterion, device)
        print(f"Accuracy after iteration {iteration} (Sparsity: {current_sparsity_percent:.2f}%): {current_accuracy:.2f}%")

        # --- Check Threshold ---
        if current_accuracy < accuracy_threshold:
            print(f"\nAccuracy ({current_accuracy:.2f}%) dropped below threshold ({accuracy_threshold:.2f}%). Stopping.")
            break
        else:
            current_save_path = os.path.join(model_save_dir, f'pruned_model_iter_{iteration}_sparsity_{current_sparsity_percent:.1f}.pth')
            save_model(model, current_save_path)
            last_good_model_path = current_save_path
            print(f"Saved model meeting threshold to: {current_save_path}")

    # --- End of Loop ---
    print("\n--- Iterative Pruning Finished ---")
    if last_good_model_path:
        print(f"The last model saved that met the accuracy threshold is: {last_good_model_path}")
        print("Loading and evaluating the final selected model...")
        final_model = SimpleCNN(num_classes=10)
        if load_model(final_model, last_good_model_path, device):
             final_accuracy, _ = evaluate_model(final_model, testloader, criterion, device)
             final_sparsity, _, _ = calculate_sparsity(final_model)
             print(f"Final Selected Model Sparsity: {final_sparsity:.2f}%")
             print(f"Final Selected Model Accuracy: {final_accuracy:.2f}%")
        else:
            print(f"Error loading the final model from {last_good_model_path}")
            last_good_model_path = None
    else:
        print("No pruned model met the accuracy threshold (or initial model was below threshold).")

    return last_good_model_path
