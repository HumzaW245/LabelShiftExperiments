
from torchvision import datasets,transforms
import torch
import numpy as np


import torchvision.transforms as transforms
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import sys
import json
from functools import partial

from WaterbirdsData.wb_data import WaterBirdsDataset, get_loader_cub, get_transform_cub

from CelebAData.celebA_data import CelebADataset, get_loader_celebA, get_transform_celebA

from WaterbirdsData.utils import Logger, AverageMeter, set_seed, evaluate, get_y_p

def getTrainTestLoaders(config):

    # Just attaching the config dict to args argument and rest of code is same as dfr repo
    args = config.spuriousConfig 

    
    assert args.reweight_groups + args.reweight_classes <= 1
    assert args.multitask + args.predict_place <= 1

    print('Preparing directory %s' % args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'command.sh'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')

    # with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
    #     args_json = json.dumps(vars(args))
    #     f.write(args_json)

    set_seed(args.seed)

    # writer = SummaryWriter(log_dir=args.output_dir)
    # logger = Logger(os.path.join(args.output_dir, 'log.txt'))

    splits = ["train", "test", "val"]

    # Data
    target_resolution = (224, 224)
    train_transform_wb = get_transform_cub(target_resolution=target_resolution, train=True, augment_data=args.augment_data, custom_data_transform = args.custom_data_transform)
    test_transform_wb = get_transform_cub(target_resolution=target_resolution, train=False, augment_data=args.augment_data, custom_data_transform = args.custom_data_transform)


    train_transform_celebA = get_transform_celebA(target_resolution=target_resolution, train=True, augment_data=args.augment_data, custom_data_transform = args.custom_data_transform)
    test_transform_celebA = get_transform_celebA(target_resolution=target_resolution, train=False, augment_data=args.augment_data, custom_data_transform = args.custom_data_transform)


    if args.spuriousDataset == 'Waterbirds':
        basedir = args.data_dir_wb
        get_loader = get_loader_cub
        trainset = WaterBirdsDataset(basedir=basedir, split="train", transform=train_transform_wb)
        testset_dict = {
            'wb': WaterBirdsDataset(basedir=args.test_wb_dir, split="test", transform=test_transform_wb),
            'wb_val': WaterBirdsDataset(basedir=args.test_wb_dir, split="val", transform=test_transform_wb),
        }

        if not args.predict_place and not (args.test_grey_dir is None):
            testset_dict['grey'] = WaterBirdsDataset(basedir=args.test_grey_dir, split="test", transform=test_transform_wb)
        if ((args.predict_place) and not (args.test_places_dir is None)) or args.multitask:
            testset_dict['places'] = WaterBirdsDataset(basedir=args.test_places_dir, split="test", transform=test_transform_wb)

    
    #'''
    #Note: The source code for DFR uses place for code convenience even for celebA instead of gender
    #This naming difference is also here where we use wb and wb_val for convenience since the original code was written
    #for waterbirds. This can be cleaned up and names of variables can be generalized later.
    #'''
    elif args.spuriousDataset == 'CelebA': 
        basedir = args.data_dir_celebA
        get_loader = get_loader_celebA
        trainset = CelebADataset(basedir=basedir, split="train", transform=train_transform_celebA)
        testset_dict = {
            'wb': CelebADataset(basedir=args.test_celebA_dir, split="test", transform=test_transform_celebA),
            'wb_val': CelebADataset(basedir=args.test_celebA_dir, split="val", transform=test_transform_celebA),
        }

    else:
        print(f'\n\n\n\n\n spuriousDataset in config.py NOT Recognized \n\n\n\n')
        assert False

    if args.num_minority_groups_remove > 0:
        print("Removing minority groups")
        print("Initial groups", np.bincount(trainset.group_array))
        group_counts = trainset.group_counts
        minority_groups = np.argsort(group_counts.numpy())[:args.num_minority_groups_remove]
        minority_groups
        idx = np.where(np.logical_and.reduce(
            [trainset.group_array != g for g in minority_groups], initial=True))[0]
        trainset.y_array = trainset.y_array[idx]
        trainset.group_array = trainset.group_array[idx]
        trainset.confounder_array = trainset.confounder_array[idx]
        trainset.filename_array = trainset.filename_array[idx]
        trainset.metadata_df = trainset.metadata_df.iloc[idx]
        print("Final groups", np.bincount(trainset.group_array))

  
    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 1, 'pin_memory': True}
    
    #Non reweighted so reweighting args are set to False
    train_loader = get_loader(
        trainset, train=True, reweight_groups=False,
        reweight_classes=False, reweight_places=False, **loader_kwargs)

    #Reweighted dataset (Only different if config is setup to use reweighting)    
    train_loader_rw = get_loader(
        trainset, train=True, reweight_groups=args.reweight_groups,
        reweight_classes=args.reweight_classes, reweight_places=args.reweight_places, **loader_kwargs)

    test_loader_dict = {}
    for test_name, testset_v in testset_dict.items():
        test_loader_dict[test_name] = get_loader(
            testset_v, train=False, reweight_groups=None,
            reweight_classes=None, reweight_places=None, **loader_kwargs)

    test_loader_dict_rw = {}
    for test_name, testset_v in testset_dict.items():
        test_loader_dict_rw[test_name] = get_loader(
            testset_v, train=False, reweight_groups=args.reweight_groups,
            reweight_classes=args.reweight_classes, reweight_places=args.reweight_places, **loader_kwargs)
    n_classes = trainset.n_classes
    print(f'Got all loaders')
    return train_loader, train_loader_rw, test_loader_dict, test_loader_dict_rw, n_classes