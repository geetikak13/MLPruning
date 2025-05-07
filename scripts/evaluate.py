# --- scripts/evaluate.py ---
import torch
import torch.nn as nn
import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import SimpleCNN
from src.data_loader import get_cifar10_loaders
from src.utils import (load_model, evaluate_model, calculate_sparsity,
                       get_parameter_count, get_non_zero_parameter_count,
                       get_device, calculate_flops)

def main(args):
    """
    Loads a saved model checkpoint, evaluates its performance, and reports metrics.
    Ensures float32 compatibility.
    """
    print("--- Evaluation Script ---"); device = get_device()
    _, testloader, _ = get_cifar10_loaders(args.batch_size, args.data_dir, num_workers=args.num_workers)

    # Initialize model architecture and ensure float32
    model = SimpleCNN(num_classes=10).float() # <<< Added .float() >>>

    print(f"Loading model from: {args.model_path}")
    # load_model utility also ensures float32 after loading state_dict
    if not load_model(model, args.model_path, device):
        print("Exiting: Failed to load model."); return

    criterion = nn.CrossEntropyLoss()
    print("\nCalculating metrics...")
    sparsity, zero_weights, total_weights = calculate_sparsity(model)
    total_params = get_parameter_count(model); non_zero_params = get_non_zero_parameter_count(model)
    macs, thop_params = calculate_flops(model, device) # calculate_flops ensures float32
    flops = macs * 2 if macs is not None else None

    print(f"\nEvaluating model performance on {device}...")
    accuracy, avg_loss = evaluate_model(model, testloader, criterion, device) # evaluate_model ensures float32

    print("\n--- Evaluation Summary ---")
    print(f"Model Path:       {args.model_path}")
    print(f"Sparsity:         {sparsity:.2f}% ({zero_weights}/{total_weights} zero weights)")
    print(f"Total Params:     {total_params:,d} (~{total_params/1e6:.2f} M)")
    print(f"Non-Zero Params:  {non_zero_params:,d} (~{non_zero_params/1e6:.2f} M)")
    if macs is not None:
        print(f"Estimated MACs:   {macs:,.0f} (~{macs/1e9:.2f} GMACs)")
        print(f"Estimated FLOPs:  ~{flops:,.0f} (~{flops/1e9:.2f} GFLOPs)")
    else: print(f"Estimated MACs:   Failed to calculate (is 'thop' installed?)")
    print(f"Test Accuracy:    {accuracy:.2f}%")
    print(f"Test Loss:        {avg_loss:.4f}")
    print("--- Evaluation Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a CIFAR-10 CNN Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint file (.pth) to evaluate.')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for CIFAR-10 data.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for data loader.')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for DataLoader.')
    args = parser.parse_args(); main(args)
