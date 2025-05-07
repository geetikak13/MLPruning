# --- src/data_loader.py ---
import torch
import torchvision
import torchvision.transforms as transforms
import ssl # Import ssl module
import urllib # Import urllib

def get_cifar10_loaders(batch_size=128, data_dir='./data', num_workers=2):
    """
    Downloads CIFAR-10 dataset and provides DataLoader objects.
    Includes a workaround for SSL certificate verification issues during download.

    Args:
        batch_size (int): Number of samples per batch.
        data_dir (str): Directory to store/load the data.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        tuple: (trainloader, testloader, classes)
               trainloader: DataLoader for the training set.
               testloader: DataLoader for the test set.
               classes: Tuple of class names.
    """
    # --- SSL Certificate Workaround ---
    # This is often needed on macOS and some environments to download datasets
    # It temporarily disables SSL certificate verification for the download process.
    # Store the original context creation function.
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # Legacy Python versions might not have this attribute, so we pass.
        pass
    else:
        # If the attribute exists, replace the default context factory with the unverified one.
        ssl._create_default_https_context = _create_unverified_https_context
    # --- End SSL Workaround ---

    # Define transformations for the training set.
    # Includes data augmentation (RandomCrop, RandomHorizontalFlip) for better generalization.
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # Randomly crop image to 32x32 after padding
        transforms.RandomHorizontalFlip(),    # Randomly flip image horizontally
        transforms.ToTensor(),                # Convert PIL Image to PyTorch Tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # Normalize with CIFAR-10 mean/std
    ])

    # Define transformations for the test set.
    # No data augmentation, only ToTensor and normalization.
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Download and load the training data using torchvision.datasets.CIFAR10
    try:
        print(f"Attempting to download/load CIFAR-10 trainset from {data_dir}...")
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir,    # Directory where data will be stored/found
            train=True,       # Load the training split
            download=True,    # Download if not present locally
            transform=transform_train # Apply training transformations
        )
        print("Trainset loaded successfully.")
    except urllib.error.URLError as e:
        # Handle potential download errors (e.g., network issues, SSL problems)
        print(f"Error downloading trainset: {e}")
        print("Please check your internet connection and SSL certificate configuration.")
        print("If the SSL error persists, ensure your Python environment's certificates are up to date.")
        raise e # Re-raise the error to stop execution if download fails
    except Exception as e:
        print(f"An unexpected error occurred loading the trainset: {e}")
        raise e

    # Create a DataLoader for the training set.
    # pin_memory=True can speed up data transfer to GPU if using CUDA.
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size, # Number of samples per batch
        shuffle=True,          # Shuffle data at every epoch
        num_workers=num_workers, # Number of subprocesses for data loading
        pin_memory=True
    )

    # Download and load the test data.
    try:
        print(f"Attempting to download/load CIFAR-10 testset from {data_dir}...")
        testset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=False,      # Load the test split
            download=True,
            transform=transform_test # Apply test transformations
        )
        print("Testset loaded successfully.")
    except urllib.error.URLError as e:
        print(f"Error downloading testset: {e}")
        print("Please check your internet connection and SSL certificate configuration.")
        raise e # Re-raise the error
    except Exception as e:
        print(f"An unexpected error occurred loading the testset: {e}")
        raise e

    # Create a DataLoader for the test set.
    # No shuffling needed for the test set.
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Define the class names for CIFAR-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Return the DataLoaders and class names
    return trainloader, testloader, classes
