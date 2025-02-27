# data_loader.py
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

def get_cifar100_loaders(data_dir, batch_size=64, num_workers=4, validation_split=0.1):
    """
    Returns DataLoaders for the CIFAR-100 dataset.

    Args:
        data_dir (str): Directory where CIFAR-100 data will be stored.
        batch_size (int): Batch size for the DataLoaders.
        num_workers (int): Number of subprocesses to use for data loading.
        validation_split (float): Fraction of training data to use for validation.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Define transformations for training and testing
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761])
    ])

    # Load the full training dataset and the test dataset
    train_dataset_full = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_test
    )

    # Split training data into training and validation sets
    total_train = len(train_dataset_full)
    val_size = int(total_train * validation_split)
    train_size = total_train - val_size
    train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

    # Create DataLoaders for each split
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    data_dir = "../data/cifar100"  # Adjust this path as needed
    train_loader, val_loader, test_loader = get_cifar100_loaders(data_dir)
    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))
