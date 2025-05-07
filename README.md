# Model Compression and Pruning Project (MSML 604)

This project implements and evaluates two heuristic numerical methods for pruning a Convolutional Neural Network (CNN) trained on CIFAR-10:
1.  **Iterative Greedy Magnitude Pruning (iGreedy Threshold):** Prunes weights iteratively based on magnitude until a target accuracy threshold is met.
2.  **L1 Regularization Pruning:** Trains the network with an L1 penalty to induce sparsity, followed by weight thresholding and optional fine-tuning.

The goal is to reduce model complexity (parameters, FLOPs) while maintaining acceptable accuracy. The code supports running on NVIDIA CUDA, Apple Silicon (MPS), or CPU automatically and includes utilities for comparison and visualization.

**Team Members:** Geetika Khanna, Archit Harsh, Himal Sharma

## Project Structure
```
mlpruning/
├── data/                     # CIFAR-10 dataset (auto-downloaded)
├── models/                   # Saved model checkpoints
│   ├── base_model.pth        # Trained base model before pruning
│   ├── pruned_threshold/     # Default output dir for iGreedy threshold models
│   └── pruned_l1reg/         # Default output dir for L1 Regularization models
├── results/                  # Directory for saving comparison plots & results
├── src/                      # Source code library
│   ├── data_loader.py        # CIFAR-10 data loading & preprocessing (with SSL fix)
│   ├── model.py              # CNN model definition
│   ├── pruning.py            # Pruning logic (iGreedy threshold method)
│   └── utils.py              # Helper functions (device handling, save/load, evaluate, metrics incl. FLOPs)
├── scripts/                  # Executable scripts
│   ├── train.py              # Script to train the base model
│   ├── prune_igreedy.py      # Script for iGreedy threshold pruning
│   ├── prune_l1reg.py        # Script for L1 Regularization pruning
│   ├── evaluate.py           # Script to evaluate a single model checkpoint (incl. FLOPs)
│   └── compare_results.py    # Script to compare and visualize results across models/methods
├── .gitignore                # Git ignore configuration
├── requirements.txt          # Python dependencies (incl. thop, pandas)
└── README.md                 # This file
```

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    git clone https://github.com/geetikak13/MLPruning
    cd model-pruning
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # Or use: python3 -m pip install -r requirements.txt
    ```
    *Note: This installs PyTorch, Torchvision, NumPy, Matplotlib, Pandas, and thop.*

## Dataset Download

The CIFAR-10 dataset is required. It will be automatically downloaded to the `./data` directory when first needed by any script. (See previous README version for manual download instructions if desired).

## Usage

The typical workflow involves training a base model, then applying one or both pruning methods, and finally comparing the results.

1.  **Train the Base Model:**
    ```bash
    python3 scripts/train.py --epochs 50 --lr 0.001 --batch_size 128 --num_workers 2
    ```
    * Saves the best model to `./models/base_model.pth`.

2.  **Apply Pruning Method 1: iGreedy Threshold:**
    ```bash
    python3 scripts/prune_igreedy.py \
        --base_model_path ./models/base_model.pth \
        --accuracy_threshold 84.0 \
        --prune_step_amount 0.2 \
        --fine_tune_epochs 5 \
        --pruned_model_dir ./models/pruned_threshold
    ```
    * **Requires** `--accuracy_threshold`.
    * Prunes iteratively until accuracy drops below the threshold.
    * Saves models meeting the threshold to the specified directory.
    * Reports the path of the last saved "good" model.

3.  **Apply Pruning Method 2: L1 Regularization:**
    ```bash
    python3 scripts/prune_l1reg.py \
        --l1_lambda 1e-5 \
        --threshold 1e-4 \
        --l1_epochs 50 \
        --fine_tune_epochs 10 \
        --final_model_save_path ./models/pruned_l1reg/final_l1_pruned_model_1e-5.pth \
        [--base_model_path ./models/base_model.pth]
    ```
    * **Requires** `--l1_lambda` (regularization strength) and `--threshold` (for post-training pruning).
    * Trains with L1 penalty, applies thresholding, fine-tunes (optional), saves final model.
    * Optionally starts from `--base_model_path` instead of training from scratch.

4.  **Evaluate a Single Model:**
    Calculates accuracy, sparsity, parameter counts, and estimated FLOPs/MACs for any saved model.
    ```bash
    # Evaluate base model
    python3 scripts/evaluate.py --model_path ./models/base_model.pth

    # Evaluate an iGreedy model
    python3 scripts/evaluate.py --model_path ./models/pruned_threshold/pruned_model_iter_5_sparsity_41.0.pth

    # Evaluate an L1 model
    python3 scripts/evaluate.py --model_path ./models/pruned_l1reg/final_l1_pruned_model_1e-5.pth
    ```

5.  **Compare and Visualize Results:**
    Compares models from the base path and specified pruning directories.
    ```bash
    python3 scripts/compare_results.py \
        --base_model_path ./models/base_model.pth \
        --igreedy_dir ./models/pruned_threshold \
        --l1reg_dir ./models/pruned_l1reg \
        --results_dir ./results
    ```
    * Generates plots (Accuracy vs. Sparsity, Accuracy vs. GFLOPs) in `./results`.
    * Prints a summary table comparing metrics for all found models.

## Customization

* **Model Architecture:** Modify `src/model.py`.
* **Pruning Parameters:** Adjust arguments in `scripts/prune_igreedy.py` or `scripts/prune_l1reg.py`.
* **Training Parameters:** Modify arguments in `scripts/train.py`.
* **Visualization:** Modify `scripts/compare_results.py`.

## `.gitignore`

A `.gitignore` file is included to prevent common Python artifacts, virtual environments, datasets (`data/`), saved models (`models/*/*.pth`), and results (`results/`) from being committed to version control.
