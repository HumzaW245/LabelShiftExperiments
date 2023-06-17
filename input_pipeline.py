import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

def _filter_to_k_shot(dataset, num_classes, k):
    keep_example = []
    class_counts = torch.zeros(num_classes, dtype=torch.int32)
    
    for _, label in dataset:
        keep = class_counts[label] < k
        keep_example.append(keep)
        if keep:
            class_counts[label] += 1
        if (class_counts == k).all():
            break
    
    indices = [i for i, keep in enumerate(keep_example) if keep]
    dataset = Subset(dataset, indices)
    
    return dataset

def create_vtab_dataset_balanced(dataset, image_size, batch_size, data_fraction):
    assert dataset in VTAB_TASKS
    num_classes = dloader.get_num_classes()
    n_shots = max(int(1000 * data_fraction / num_classes), 1)
    logging.info('n_shots: %d', n_shots)
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((-1.0,), (2.0,))
    ])
    
    dataset = ImageFolder(root='path/to/dataset/trainval', transform=transform)
    dataset = _filter_to_k_shot(dataset, num_classes, n_shots)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def create_vtab_dataset(dataset, image_size, batch_size, mode, eval_mode='test', valid_fold_id=4):
    assert 0 <= valid_fold_id < 5
    if mode not in ('train', 'eval'):
        raise ValueError("mode should be 'train' or 'eval'")
    is_training = mode == 'train'
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((-1.0,), (2.0,))
    ])
    
    if eval_mode == 'test':
        split_name = 'train800val200' if is_training else 'test'
    elif eval_mode == 'valid':
        val_start, val_end = valid_fold_id * 200, (valid_fold_id + 1) * 200
        if is_training:
            split_name = f'train[:{val_start}]+train[{val_end}:1000]'
        else:
            split_name = f'train[{val_start}:{val_end}]'
        logging.info('Using split_name: %s', split_name)
    else:
        raise ValueError(f'eval_mode: {eval_mode} invalid')
    
    dataset = ImageFolder(root='path/to/dataset/trainval', transform=transform)
    
    if is_training:
        indices = [i for i, (_, label) in enumerate(dataset) if i < val_start or i >= val_end]
        dataset = Subset(dataset, indices)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader
