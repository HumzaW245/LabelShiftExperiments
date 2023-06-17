import functools
import logging
import torch
import torchvision
from torch.utils.data import DataLoader

def _filter_to_k_shot(dataset, num_classes, k):
    """Filters k-shot subset from a dataset."""
    # Indices of included examples in the k-shot balanced dataset.
    keep_example = []
    # Keep track of the number of examples per class included in `keep_example`.
    class_counts = torch.zeros([num_classes], dtype=torch.int32)
    
    for _, label in dataset:
        # If there are less than `k` examples of class `label` in `example_indices`,
        # keep this example and update the class counts.
        keep = class_counts[label] < k
        keep_example.append(keep)
        if keep:
            class_counts[label] += 1
        # When there are `k` examples for each class included in `keep_example`,
        # stop searching.
        if (class_counts == k).all():
            break
    
    return torch.utils.data.Subset(dataset, torch.nonzero(torch.tensor(keep_example)).squeeze())

def create_vtab_dataset_balanced(dataset, image_size, batch_size, data_fraction):
    """Creates a VTAB dataset using torchvision.datasets for k-shot learning.

    Args:
        dataset: torchvision.datasets.Dataset, the dataset to use
        image_size: int, size of the input images
        batch_size: int, batch size for the data loaders
        data_fraction: float, used to calculate the number of shots per class

    Returns:
        train_loader, test_loader: DataLoaders for the k-shot balanced dataset
    """
    num_classes = len(dataset.classes)
    train_dataset = dataset(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    test_dataset = dataset(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

    n_shots = max(int(len(1000) * data_fraction / num_classes), 1)
    logging.info('n_shots: %d', n_shots)

    filtered_train_dataset = _filter_to_k_shot(train_dataset, num_classes, n_shots)
    filtered_train_loader = DataLoader(filtered_train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return filtered_train_loader, test_loader

def create_vtab_dataset(dataset, image_size, batch_size, mode, eval_mode='test', valid_fold_id=4):
    """Creates a VTAB dataset using torchvision.datasets for training or evaluation.

    Args:
        dataset: torchvision.datasets.Dataset, the dataset to use
        image_size: int, size of the input images
        batch_size: int, batch size for the data loaders
        mode: str, whether to build the input function for training or evaluation ('train' or 'eval')
        eval_mode: str, whether to build the input functions for validation or test runs ('valid' or 'test')
        valid_fold_id: int, valid fold ID for validation mode

    Returns:
        train_loader, eval_loader: DataLoaders for the VTAB dataset
    """
    train_dataset = dataset(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())

    if mode == 'train':
        if eval_mode == 'valid':
            val_start, val_end = valid_fold_id * 200, (valid_fold_id + 1) * 200
            train_indices = list(range(val_start)) + list(range(val_end, len(train_dataset)))
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        return train_loader
    else:
        if eval_mode == 'valid':
            val_start, val_end = valid_fold_id * 200, (valid_fold_id + 1) * 200
            eval_indices = list(range(val_start, val_end))
            eval_dataset = torch.utils.data.Subset(train_dataset, eval_indices)
        else:
            eval_dataset = dataset(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        return eval_loader
