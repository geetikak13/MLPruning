# --- scripts/compare_results.py ---
import torch
import torch.nn as nn
import os
import sys
import argparse
import matplotlib.pyplot as plt
import glob
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import SimpleCNN
from src.data_loader import get_cifar10_loaders
from src.utils import (load_model, evaluate_model, calculate_sparsity,
                       get_non_zero_parameter_count, get_device, calculate_flops)

try: from tabulate import tabulate; TABULATE_AVAILABLE = True
except ImportError: TABULATE_AVAILABLE = False; print("Optional: 'pip install tabulate' for better table formatting.")

def evaluate_single_model(model_path, device, batch_size, data_dir, num_workers, model_type):
    """Loads, evaluates a model, and returns metrics. Ensures float32."""
    print(f"\n--- Evaluating [{model_type}]: {os.path.basename(model_path)} on {device} ---")
    # Initialize model and ensure float32
    model = SimpleCNN(num_classes=10).float() # <<< Added .float() >>>
    if not load_model(model, model_path, device): # load_model also ensures float32
        print(f"Skipping {model_path} due to loading error."); return None
    _, testloader, _ = get_cifar10_loaders(batch_size, data_dir, num_workers=num_workers); criterion = nn.CrossEntropyLoss()
    accuracy, loss = evaluate_model(model, testloader, criterion, device) # evaluate_model ensures float32
    sparsity, _, _ = calculate_sparsity(model)
    non_zero_params = get_non_zero_parameter_count(model)
    macs, _ = calculate_flops(model, device) # calculate_flops ensures float32
    flops = macs * 2 if macs is not None else None
    return {'path': model_path, 'name': os.path.basename(model_path), 'type': model_type, 'accuracy': accuracy, 'loss': loss, 'sparsity': sparsity, 'non_zero_params': non_zero_params, 'macs': macs, 'flops': flops}

def plot_results(df_results, save_dir):
    """Generates and saves comparison plots."""
    if df_results.empty: print("No results to plot."); return
    os.makedirs(save_dir, exist_ok=True); plt.style.use('seaborn-v0_8-whitegrid')
    df_base = df_results[df_results['type'] == 'Base']
    df_igreedy = df_results[df_results['type'] == 'iGreedy'].sort_values('sparsity')
    df_l1 = df_results[df_results['type'] == 'L1Reg'].sort_values('sparsity')

    # Plot 1: Accuracy vs. Sparsity
    plt.figure(figsize=(10, 6))
    if not df_base.empty: plt.scatter(df_base['sparsity'], df_base['accuracy'], marker='*', s=200, color='red', label='Base Model', zorder=5)
    if not df_igreedy.empty: plt.plot(df_igreedy['sparsity'], df_igreedy['accuracy'], marker='o', linestyle='--', label='iGreedy Threshold Steps')
    if not df_l1.empty: plt.scatter(df_l1['sparsity'], df_l1['accuracy'], marker='^', s=100, color='green', label='L1 Regularization Result(s)', zorder=4)
    plt.title('Model Accuracy vs. Sparsity'); plt.xlabel('Sparsity (%)'); plt.ylabel('Test Accuracy (%)')
    plt.legend(); plt.grid(True); plot_path1 = os.path.join(save_dir, 'accuracy_vs_sparsity.png')
    plt.savefig(plot_path1); print(f"Saved plot: {plot_path1}"); plt.close()

    # Plot 2: Accuracy vs. FLOPs
    df_plot_flops = df_results.dropna(subset=['flops']).copy()
    if 'flops' in df_plot_flops.columns: df_plot_flops['gflops'] = df_plot_flops['flops'] / 1e9
    else: print("Warning: 'flops' column not found."); return
    df_base_flops = df_plot_flops[df_plot_flops['type'] == 'Base']
    df_igreedy_flops = df_plot_flops[df_plot_flops['type'] == 'iGreedy'].sort_values('sparsity')
    df_l1_flops = df_plot_flops[df_plot_flops['type'] == 'L1Reg'].sort_values('sparsity')
    plt.figure(figsize=(10, 6))
    if not df_base_flops.empty: plt.scatter(df_base_flops['gflops'], df_base_flops['accuracy'], marker='*', s=200, color='red', label='Base Model', zorder=5)
    if not df_igreedy_flops.empty:
        if df_igreedy_flops['gflops'].nunique() > 1: plt.plot(df_igreedy_flops['gflops'], df_igreedy_flops['accuracy'], marker='o', linestyle='--', label='iGreedy Threshold Steps')
        else: plt.scatter(df_igreedy_flops['gflops'], df_igreedy_flops['accuracy'], marker='o', label='iGreedy Threshold Steps (Constant FLOPs)')
    if not df_l1_flops.empty: plt.scatter(df_l1_flops['gflops'], df_l1_flops['accuracy'], marker='^', s=100, color='green', label='L1 Regularization Result(s)', zorder=4)
    plt.title('Model Accuracy vs. Estimated GFLOPs'); plt.xlabel('Estimated GFLOPs'); plt.ylabel('Test Accuracy (%)')
    plt.legend(); plt.grid(True)
    if not df_igreedy_flops.empty and df_igreedy_flops['gflops'].nunique() == 1: plt.text(0.95, 0.01, '*FLOPs constant due to unstructured pruning', verticalalignment='bottom', horizontalalignment='right', transform=plt.gca().transAxes, color='gray', fontsize=8)
    plot_path2 = os.path.join(save_dir, 'accuracy_vs_gflops.png'); plt.savefig(plot_path2); print(f"Saved plot: {plot_path2}"); plt.close()

def print_summary_table(df_results):
    """Prints a formatted summary table."""
    if df_results.empty: print("No results to display."); return
    df_results_sorted = df_results.sort_values(by=['type', 'sparsity'])
    print("\n--- Results Summary ---"); headers = ["Type", "Name", "Accuracy (%)", "Sparsity (%)", "Non-Zero Params", "GFLOPs"]
    data_rows = []
    for _, r in df_results_sorted.iterrows():
        flops_str = f"{r['flops']/1e9:.2f}" if pd.notna(r['flops']) else "N/A"
        params_str = f"{r['non_zero_params']:,d}"; acc_str = f"{r['accuracy']:.2f}"; sparsity_str = f"{r['sparsity']:.2f}"
        name_str = r['name'] if len(r['name']) < 38 else r['name'][:35] + "..."
        data_rows.append([r['type'], name_str, acc_str, sparsity_str, params_str, flops_str])
    if TABULATE_AVAILABLE: print(tabulate(data_rows, headers=headers, tablefmt="grid"))
    else:
        header_line = f"{headers[0]:<10} | {headers[1]:<40} | {headers[2]:<12} | {headers[3]:<12} | {headers[4]:<15} | {headers[5]:<8}"; print(header_line); print("-" * len(header_line))
        for row in data_rows: print(f"{row[0]:<10} | {row[1]:<40} | {row[2]:<12} | {row[3]:<12} | {row[4]:<15} | {row[5]:<8}")
        print("-" * len(header_line))

def main(args):
    """Main function to compare models."""
    device = get_device(); all_results_data = []
    if args.base_model_path and os.path.exists(args.base_model_path):
        print("Evaluating Base Model..."); base_result = evaluate_single_model(args.base_model_path, device, args.batch_size, args.data_dir, args.num_workers, "Base")
        if base_result: all_results_data.append(base_result)
    else: print(f"Warning: Base model not found: {args.base_model_path}")
    if args.igreedy_dir and os.path.isdir(args.igreedy_dir):
        print(f"\nEvaluating iGreedy Models from: {args.igreedy_dir}..."); igreedy_files = glob.glob(os.path.join(args.igreedy_dir, '*.pth'))
        if not igreedy_files: print(f"Warning: No '.pth' files found in {args.igreedy_dir}")
        else:
            for model_file in sorted(igreedy_files):
                result = evaluate_single_model(model_file, device, args.batch_size, args.data_dir, args.num_workers, "iGreedy");
                if result: all_results_data.append(result)
    else: print(f"Warning: iGreedy directory not found: {args.igreedy_dir}")
    if args.l1reg_dir and os.path.isdir(args.l1reg_dir):
        print(f"\nEvaluating L1Reg Models from: {args.l1reg_dir}..."); l1reg_files = glob.glob(os.path.join(args.l1reg_dir, '*.pth'))
        if not l1reg_files: print(f"Warning: No '.pth' files found in {args.l1reg_dir}")
        else:
            for model_file in sorted(l1reg_files):
                 base_name = os.path.basename(model_file)
                 if "_temp.pth" not in base_name and "_thresholded_only.pth" not in base_name:
                    result = evaluate_single_model(model_file, device, args.batch_size, args.data_dir, args.num_workers, "L1Reg");
                    if result: all_results_data.append(result)
    else: print(f"Warning: L1Reg directory not found: {args.l1reg_dir}")
    df_results = pd.DataFrame(all_results_data); print_summary_table(df_results); plot_results(df_results, args.results_dir)
    print("\n--- Comparison Script Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare Base and Pruned Model Results')
    parser.add_argument('--base_model_path', type=str, default='./models/base_model.pth', help='Path to the pre-trained base model file.')
    parser.add_argument('--igreedy_dir', type=str, default='./models/pruned_threshold', help='Directory containing iGreedy threshold pruned models.')
    parser.add_argument('--l1reg_dir', type=str, default='./models/pruned_l1reg', help='Directory containing L1 regularization pruned models.')
    parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save comparison plots and results.')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for CIFAR-10 data.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation data loader.')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for DataLoader.')
    args = parser.parse_args(); main(args)
