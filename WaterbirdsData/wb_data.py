import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, BatchSampler
from torch.utils.data.sampler import WeightedRandomSampler

from WaterbirdsData.data_transforms import AugWaterbirdsCelebATransform


class BalancedBatchSampler(BatchSampler):
    def __init__(self, dataset, reweight_classes=False, reweight_groups=False, reweight_places=False, batch_size=0):
        self.dataset = dataset
        self.reweight_classes = reweight_classes
        self.reweight_groups = reweight_groups
        self.reweight_places = reweight_places
        self.batch_size = batch_size

        self.indices_per_class, self.indices_per_group, self.indices_per_place = self._get_indices_per_X()
        self.total_samples = len(self.dataset)
        self.num_batches = self.total_samples // self.batch_size

    def _get_indices_per_X(self):
        # Implementation for getting indices for e.g. per group as a dictionary with key being the group and value being the list of indexes of all data for that group
        indices_per_class = {}
        indices_per_group = {}
        indices_per_place = {}

        #Store indices into corresponding class, group and place
        for idx, item in enumerate(self.dataset):
            
            # SEE get_item of dataset... img, y, g, p (input, target, group, place)
            #data = batch[0] //Don't need data, just want indices of target, group, place for when balancing based on 1 of these
            target = item[1] #  Classes is target
            group = item[2] 
            place = item[3]

            #Store index to its corresponding class in the indices_per_class dictionary which has keys = class, values = indices with that class in dataset
            if target not in indices_per_class:
                indices_per_class[target] = []
            indices_per_class[target].append(idx)

            #Store index to its corresponding group in the indices_per_group dictionary which has keys = group, values = indices with that group in dataset
            if group not in indices_per_group:
                indices_per_group[group] = []
            indices_per_group[group].append(idx)

            #Store index to its corresponding place in the indices_per_place dictionary which has keys = place, values = indices with that place in dataset
            if place not in indices_per_place:
                indices_per_place[place] = []
            indices_per_place[place].append(idx)


        return (indices_per_class, indices_per_group, indices_per_place)

    '''
    _get_indices_per_X has all the indices perX (e.g. all indices per group are returned)

    This is just to generate a batch of them every time a batch needs to be returned (yielded in __iter__) 

    ************IF DONT WANT TO REUSE SAMPLES WHEN MAKING BATCHES, after batch_indices.extend(indicesToAddFromX) in the for loop in each if/elif, put a function to remove(indicesToAddFromX from self.indices_per_group)
    '''
    def _get_batch_indices_per_X(self):
        batch_indices = []
        if self.reweight_groups:
            # Select samples from balanced groups
            for group, indices_list in self.indices_per_group.items():
                indices = torch.tensor(indices_list)
                indicesToAddFromGroup = indices[torch.randperm(len(indices))[:self.batch_size // len(self.indices_per_group)]].tolist() #Divide batch size by number of categories so e.g. 4 groups and 128 batch size then divide 128/4 = 32 per group
                #print(f'These are how many indices there are in total for group {group} being added to the batch_indices variable= {len(indicesToAddFromGroup)} OUT OF TOTAL INDICES FOR GROUP ={len(indices)}')
                batch_indices.extend(indicesToAddFromGroup)

        elif self.reweight_classes:
            # Select samples from balanced classes
            for target, indices_list in self.indices_per_class.items():
                indices = torch.tensor(indices_list)
                indicesToAddFromClass = indices[torch.randperm(len(indices))[:self.batch_size // len(self.indices_per_class)]].tolist() #Divide batch size by number of categories so e.g. 4 CLASSes and 128 batch size then divide 128/4 = 32 per class
                #print(f'These are how many indices there are in total for class {target} being added to the batch_indices variable= {len(indicesToAddFromClass)} OUT OF TOTAL INDICES FOR CLASS ={len(indices)}')
                batch_indices.extend(indicesToAddFromClass)

        elif self.reweight_places:
            # Select samples from balanced places
            for place, indices_list in self.indices_per_place.items():
                indices = torch.tensor(indices_list)
                indicesToAddFromPlace = indices[torch.randperm(len(indices))[:self.batch_size // len(self.indices_per_place)]].tolist() #Divide batch size by number of categories so e.g. 4 places and 128 batch size then divide 128/4 = 32 per place
                #print(f'These are how many indices there are in total for place {place} being added to the batch_indices variable= {len(indicesToAddFromPlace)} OUT OF TOTAL INDICES FOR PLACE ={len(indices)}')
                batch_indices.extend(indicesToAddFromPlace)

        else:
            # Select samples without reweighting
            batch_indices = torch.randperm(self.total_samples).tolist()

        return batch_indices
        

    def __iter__(self):
        if self.reweight_groups or self.reweight_classes or self.reweight_places:

            for i in range(0, self.total_samples, self.batch_size):
                batch_indices = self._get_batch_indices_per_X()
                batchToReturn = batch_indices
                
                #print(f'\n\nbatch PULLED with batch_indices len = {len(batch_indices)} and batchToReturn = {batchToReturn}\n\n')
                yield batchToReturn
        else:
            batch_indices = self._get_batch_indices_per_X()
            for i in range(0, self.total_samples, self.batch_size):
                batchToReturn = batch_indices[i:i+self.batch_size]
                #print(f'\n\nbatch PULLED with batch_indices len = {len(batch_indices)} and batchToReturn = {batchToReturn}\n\n')
                yield batchToReturn

        

    def __len__(self):
        return self.num_batches

class WaterBirdsDataset(Dataset):
    def __init__(self, basedir, split="train", transform=None):
        try:
            split_i = ["train", "val", "test"].index(split)
        except ValueError:
            raise(f"Unknown split {split}")
        metadata_df = pd.read_csv(os.path.join(basedir, "metadata.csv"))
        self.metadata_df = metadata_df[metadata_df["split"] == split_i]
        self.basedir = basedir
        self.transform = transform
        self.y_array = self.metadata_df['y'].values
        self.p_array = self.metadata_df['place'].values
        self.n_classes = np.unique(self.y_array).size
        self.confounder_array = self.metadata_df['place'].values
        self.n_places = np.unique(self.confounder_array).size
        self.group_array = (self.y_array * self.n_places + self.confounder_array).astype('int')
        self.n_groups = self.n_classes * self.n_places
        self.group_counts = (
                torch.arange(self.n_groups).unsqueeze(1) == torch.from_numpy(self.group_array)).sum(1).float()
        self.y_counts = (
                torch.arange(self.n_classes).unsqueeze(1) == torch.from_numpy(self.y_array)).sum(1).float()
        self.p_counts = (
                torch.arange(self.n_places).unsqueeze(1) == torch.from_numpy(self.p_array)).sum(1).float()
        self.filename_array = self.metadata_df['img_filename'].values

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        p = self.confounder_array[idx]

        img_path = os.path.join(self.basedir, self.filename_array[idx])
        img = Image.open(img_path).convert('RGB')
        # img = read_image(img_path)
        # img = img.float() / 255.

        if self.transform:
            img = self.transform(img)
        return img, y, g, p


def get_transform_cub(target_resolution, train, augment_data, custom_data_transform):

    if custom_data_transform == "AugWaterbirdsCelebATransform":
        print(f'\n\n USING CUSTOM DATA TRANSFORM {custom_data_transform}\n\n')
        transform = AugWaterbirdsCelebATransform(train)
    else:
        print(f'\n\n USING DEFAULT DATA TRANSFORM')
        
        scale = 256.0 / 224.0

        if (not train) or (not augment_data):
            # Resizes the image to a slightly larger square then crops the center.
            transform = transforms.Compose([
                transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
                transforms.CenterCrop(target_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    target_resolution,
                    scale=(0.7, 1.0),
                    ratio=(0.75, 1.3333333333333333),
                    interpolation=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    return transform


def get_loader_cub(data, train, reweight_groups, reweight_classes, reweight_places, **kwargs):
    batch_size = kwargs['batch_size']
    num_workers = kwargs['num_workers']
    pin_memory = kwargs['pin_memory']

    loader = DataLoader(
        data,
        batch_sampler=BalancedBatchSampler(data, reweight_classes=reweight_classes, reweight_groups=reweight_groups, reweight_places=reweight_places, batch_size=batch_size),
        num_workers=num_workers,
        pin_memory=pin_memory
        )
    return loader


