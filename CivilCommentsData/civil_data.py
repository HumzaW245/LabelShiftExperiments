

NEED TO DEAL WITH has_wilds PART IN CLASS BaseWildsDataset

HOW TO FEED JUST THE EXCEL FILE????

NEED TO DEAL WITH has_wilds PART IN CLASS BaseWildsDataset

HOW TO FEED JUST THE EXCEL FILE????

NEED TO DEAL WITH has_wilds PART IN CLASS BaseWildsDataset

HOW TO FEED JUST THE EXCEL FILE????

NEED TO DEAL WITH has_wilds PART IN CLASS BaseWildsDataset

HOW TO FEED JUST THE EXCEL FILE????





import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, BatchSampler
from torch.utils.data.sampler import WeightedRandomSampler

from CivilCommentsData.data_transforms import BertTokenizeTransform


#------------------------------------------------NOTE--------------------------------------------------------------------------------------------
#For convenience, since metadata also uses place instead of renaming to gender and because rest of codebase is setup with waterbird variable names, using place for genders here too
#------------------------------------------------NOTE--------------------------------------------------------------------------------------------




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
            if idx % 10000 == 0:
                print(f'Index in get_indices_per_X is {idx} and length of dadtaset is {len(self.dataset)}')
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
                indicesToAddFromplace = indices[torch.randperm(len(indices))[:self.batch_size // len(self.indices_per_place)]].tolist() #Divide batch size by number of categories so e.g. 4 places and 128 batch size then divide 128/4 = 32 per place
                #print(f'These are how many indices there are in total for place {place} being added to the batch_indices variable= {len(indicesToAddFromplace)} OUT OF TOTAL INDICES FOR place ={len(indices)}')
                batch_indices.extend(indicesToAddFromplace)

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


def get_transform_civil(target_resolution, train, augment_data, custom_data_transform):

    if custom_data_transform == "BertTokenizeTransform":
        print(f'\n\n USING CUSTOM DATA TRANSFORM {custom_data_transform}\n\n')
        transform = BertTokenizeTransform(train)
    else:
        print('******ERRORRR*********NEED TO SPECIFY custom_data_transform as BertTokenizeTransform SINCE LANGUAGE DATA TO USE WITH BERT')
        assert False
    return transform


def get_loader_civil(data, train, reweight_groups, reweight_classes, reweight_places, **kwargs):
    batch_size = kwargs['batch_size']
    num_workers = kwargs['num_workers']
    pin_memory = kwargs['pin_memory']
    print(f'before getting loader')
    loader = DataLoader(
        data,
        batch_sampler=BalancedBatchSampler(data, reweight_classes=reweight_classes, reweight_groups=reweight_groups, reweight_places=reweight_places, batch_size=batch_size),
        num_workers=num_workers,
        pin_memory=pin_memory
        )
    print(f'after getting loader')
    return loader



'''

Code below repurposed from https://github.com/izmailovpavel/spurious_feature_learning/blob/main/data/datasets.py

'''

try:
    import wilds
    from wilds.datasets.wilds_dataset import WILDSSubset
    has_wilds = True
except:
    has_wilds = False

def _get_split(split):
    try:
        return ["train", "val", "test"].index(split)
    except ValueError:
        raise(f"Unknown split {split}")

def _cast_int(arr):
    if isinstance(arr, np.ndarray):
        return arr.astype(int)
    elif isinstance(arr, torch.Tensor):
        return arr.int()
    else:
        raise NotImplementedError

class SpuriousCorrelationDataset(Dataset):
    def __init__(self, basedir, split="train", transform=None):
        self.basedir = basedir
        self.metadata_df = self._get_metadata(split)
        
        self.transform = transform
        self.y_array = self.metadata_df["y"].values
        if "spurious" in self.metadata_df:
            self.spurious_array = self.metadata_df["spurious"].values
        else:
            self.spurious_array = self.metadata_df["place"].values
        self._count_attributes()
        if "group" in self.metadata_df:
            self.group_array = self.metadata_df["group"].values
        else:
            self._get_class_spurious_groups()
        self._count_groups()
        self.text = not "img_filename" in self.metadata_df
        if self.text:
            print("NLP dataset")
            self.text_array = list(pd.read_csv(os.path.join(
                basedir, "text.csv"))["text"])
        else:
            self.filename_array = self.metadata_df["img_filename"].values

    def _get_metadata(self, split):
        split_i = _get_split(split)
        metadata_df = pd.read_csv(os.path.join(self.basedir, "metadata.csv"))
        metadata_df = metadata_df[metadata_df["split"] == split_i]
        return metadata_df

    def _count_attributes(self):
        self.n_classes = np.unique(self.y_array).size
        self.n_spurious = np.unique(self.spurious_array).size
        self.y_counts = self._bincount_array_as_tensor(self.y_array)
        self.spurious_counts = self._bincount_array_as_tensor(
            self.spurious_array)

    def _count_groups(self):
        self.group_counts = self._bincount_array_as_tensor(self.group_array)
        # self.n_groups = np.unique(self.group_array).size
        self.n_groups = len(self.group_counts)

    def _get_class_spurious_groups(self):
        self.group_array = _cast_int(
            self.y_array * self.n_spurious + self.spurious_array)
        
    @staticmethod
    def _bincount_array_as_tensor(arr):
        return torch.from_numpy(np.bincount(arr)).long()

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        s = self.spurious_array[idx]
        if self.text:
            x = self._text_getitem(idx)
        else:
            x = self._image_getitem(idx)
        return x, y, g, s

    def _image_getitem(self, idx):
        img_path = os.path.join(self.basedir, self.filename_array[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def _text_getitem(self, idx):
        text = self.text_array[idx]
        if self.transform:
            text = self.transform(text)
        return text


class BaseWildsDataset(SpuriousCorrelationDataset):
    def __init__(
        self, ds_name, basedir, split, transform, y_name, spurious_name
    ):
        assert has_wilds, "wilds package not found"
        self.basedir = basedir
        self.root_dir = "/".join(self.basedir.split("/")[:-2])
        base_dataset = wilds.get_dataset(
            dataset=ds_name, download=False, root_dir=self.root_dir)
        self.dataset = base_dataset.get_subset(split, transform=transform)

        column_names = self.dataset.metadata_fields
        if y_name:
            y_idx = column_names.index(y_name)
            self.y_array = self.dataset.metadata_array[:, y_idx]
        if spurious_name:
            s_idx = column_names.index(spurious_name)
            self.spurious_idx = s_idx
            self.spurious_array = self.dataset.metadata_array[:, s_idx]
        if y_name and spurious_name:
            self._count_attributes()

    def __getitem__(self, idx):
        x, y, metadata = self.dataset[idx]
        s = metadata[self.spurious_idx]
        return x, y, s, s

    def __len__(self):
        return len(self.dataset)


class WildsCivilCommentsCoarse(BaseWildsDataset):
    def __init__(self, basedir, split="train", transform=None):
        super().__init__("civilcomments", basedir, split, transform, "y", None)
        attributes = ["male", "female", "LGBTQ", "black", "white", "christian",
                      "muslim", "other_religions"]
        column_names = self.dataset.metadata_fields
        self.spurious_cols = [column_names.index(a) for a in attributes]
        self.spurious_array = self.get_spurious(self.dataset.metadata_array)
        self._count_attributes()
        self._get_class_spurious_groups()
        self._count_groups()

    def get_spurious(self, metadata):
        if len(metadata.shape) == 1:
            return metadata[self.spurious_cols].sum(-1).clip(max=1)
        else:
            return metadata[:, self.spurious_cols].sum(-1).clip(max=1)

    def __getitem__(self, idx):
        x, y, metadata = self.dataset[idx]
        s = self.get_spurious(metadata)
        g = y * self.n_spurious + s
        return x, y, g, s
