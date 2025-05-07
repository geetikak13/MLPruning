# --- src/utils.py ---
import torch
import os
import numpy as np
try:
    # Attempt to import thop for FLOPs calculation
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    # If thop is not installed, set a flag and print a warning
    print("Warning: 'thop' library not found. FLOPs calculation will be skipped.")
    print("Install thop using: pip install thop")
    THOP_AVAILABLE = False

def get_device():
    """
    Determines and returns the best available PyTorch device (MPS, CUDA, or CPU).
    Prints the device being used.

    Returns:
        torch.device: The selected device object.
    """
    if torch.backends.mps.is_available():
        # Check if MPS (Apple Silicon GPU) is available and built
        device = torch.device("mps")
        print("Using Apple Silicon MPS device.")
    elif torch.cuda.is_available():
        # Check if CUDA (NVIDIA GPU) is available
        device = torch.device("cuda")
        print("Using NVIDIA CUDA device.")
    else:
        # Fallback to CPU
        device = torch.device("cpu")
        print("Using CPU device.")
    return device

def save_model(model, path):
    """
    Saves the model's state dictionary to the specified path.
    Ensures the model is moved to CPU before saving to avoid device-specific info.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): The file path to save the model state dictionary to.
    """
    print(f"Saving model to {path}...")
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Move model to CPU before saving state_dict to ensure compatibility
    torch.save(model.to('cpu').state_dict(), path)
    print("Model saved.")

def load_model(model, path, device):
    """
    Loads a model state dictionary from a file onto the specified device.
    Ensures the loaded model is converted to float32.
    Tries strict loading first, falls back to non-strict if needed.

    Args:
        model (torch.nn.Module): The model instance to load the state into.
        path (str): The file path of the saved state dictionary.
        device (torch.device): The target device to load the model onto.

    Returns:
        bool: True if loading was successful, False otherwise.
    """
    print(f"Loading model from {path}...")
    if not os.path.exists(path):
        print(f"Error: Model file not found at {path}")
        return False
    try:
        # Load the state dict initially to CPU for broader compatibility
        state_dict = torch.load(path, map_location='cpu')

        # Attempt strict loading first (preferred)
        model.load_state_dict(state_dict, strict=True)
        # Ensure model is float32 before moving to device
        model = model.float()
        model.to(device) # Move the model to the target device AFTER loading state
        print(f"Model loaded successfully (strict), converted to float32, and moved onto {device}.")
        return True
    except RuntimeError as e:
        # If strict loading fails (often due to pruning masks/orig keys missing or extra)
        print(f"Strict loading failed: {e}. Attempting non-strict loading...")
        try:
            # Reload using the original state_dict but with strict=False
            model.load_state_dict(state_dict, strict=False)
            # Ensure model is float32 before moving to device
            model = model.float()
            model.to(device)
            print(f"Model loaded non-strictly, converted to float32, and moved onto {device}.")
            return True
        except Exception as e2:
             # If even non-strict loading fails, report the error
             print(f"Non-strict loading also failed: {e2}")
             return False
    except Exception as e:
         # Catch any other potential loading errors
         print(f"Error loading model state_dict: {e}")
         return False


def evaluate_model(model, testloader, criterion, device):
    """
    Evaluates the model's performance on the test dataset.
    Ensures model is float32 before moving to device.

    Args:
        model (torch.nn.Module): The model to evaluate.
        testloader (torch.utils.data.DataLoader): DataLoader for the test set.
        criterion: The loss function (e.g., nn.CrossEntropyLoss).
        device (torch.device): The device to perform evaluation on.

    Returns:
        tuple: (accuracy, average_loss)
               accuracy (float): The classification accuracy percentage.
               average_loss (float): The average loss over the test set.
    """
    # <<< Ensure model is float32 BEFORE moving to device >>>
    model = model.float()
    model.to(device)
    model.eval() # Set model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0
    # Disable gradient calculations during evaluation
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # Move data to the evaluation device (input data is usually float32 by default)
            inputs, targets = inputs.to(device), targets.to(device)
            # Forward pass
            outputs = model(inputs)
            # Calculate loss
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            # Get predictions
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Calculate average loss and accuracy
    num_batches = len(testloader)
    avg_loss = test_loss / num_batches if num_batches > 0 else 0
    accuracy = 100. * correct / total if total > 0 else 0

    print(f'Test Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}% ({correct}/{total})')
    return accuracy, avg_loss

def calculate_sparsity(model):
    """
    Calculates the percentage of zero-valued weights in the model.
    Considers only parameters typically pruned (Conv2d, Linear weights).

    Args:
        model (torch.nn.Module): The model to analyze.

    Returns:
        tuple: (sparsity_percentage, zero_weights, total_weights)
    """
    total_weights = 0; zero_weights = 0
    model_on_cpu = model.to('cpu') # Calculate on CPU
    for name, param in model_on_cpu.named_parameters():
        is_weight = 'weight' in name
        module_name = name.rsplit('.', 1)[0]
        try:
            module = model_on_cpu.get_submodule(module_name)
            is_conv_or_linear = isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
        except AttributeError: is_conv_or_linear = False
        if is_weight and is_conv_or_linear:
            total_weights += param.nelement()
            zero_weights += torch.sum(param == 0).item()
    if total_weights == 0: print("Model Sparsity: No weights found."); return 0.0, 0, 0
    sparsity = 100. * zero_weights / total_weights
    print(f"Model Sparsity: {sparsity:.2f}% ({zero_weights}/{total_weights} zero weights)")
    return sparsity, zero_weights, total_weights

def get_parameter_count(model):
    """Calculates the total number of trainable parameters in the model."""
    model_on_cpu = model.to('cpu'); return sum(p.numel() for p in model_on_cpu.parameters() if p.requires_grad)

def get_non_zero_parameter_count(model):
    """Calculates the total number of non-zero trainable parameters in the model."""
    model_on_cpu = model.to('cpu'); count = 0
    for param in model_on_cpu.parameters():
        if param.requires_grad: count += torch.count_nonzero(param).item()
    return count

def calculate_flops(model, device, input_size=(1, 3, 32, 32)):
    """Calculates estimated FLOPs (or MACs) for the model using 'thop'."""
    if not THOP_AVAILABLE: return None, None
    model = model.float() # Ensure float32 for thop compatibility
    model.to(device); model.eval()
    dummy_input = torch.randn(input_size).to(device).float() # Ensure input is float32
    try:
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        gmacs = macs / 1e9; mparams = params / 1e6
        print(f"Estimated MACs (thop): {gmacs:.2f} G")
        print(f"Total Parameters (thop): {mparams:.2f} M")
        return macs, params
    except Exception as e:
        print(f"Could not calculate FLOPs/MACs using thop: {e}")
        return None, None
