
from torchvision import datasets,transforms
import torch
import numpy as np


import torchvision.transforms as transforms
from torchvision.datasets import SVHN

def getTrainTestLoaders(config):

    if config.dataset == 'SVHN':
        # Define the transforms to apply (--------------------------------------PREPROCESSING FOR EACH DATASET WHAT IS BEST TO GO WITH IMAGENETR50)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create an instance of the SVHN dataset with the transforms
        trainData = datasets.SVHN('../data', split='train', download=True, transform=transform)

        testData = datasets.SVHN('../data', split='test', download=True, transform=transform)


        learningConfig = config.learning

        train_loader = torch.utils.data.DataLoader(trainData,
                                                batch_size=learningConfig.train_batch_size,
                                                shuffle=True,
                                                drop_last=True)

        test_loader = torch.utils.data.DataLoader(testData,
                                                batch_size=learningConfig.eval_batch_size,
                                                shuffle=True,
                                                drop_last=True) #Drop last is really just to make sure if last batch is not of equal size, drop it. Nothing to do with setting it aside for testing
        return train_loader, test_loader

    elif config.dataset == 'Flowers102':
        # Define the transforms to apply
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create an instance of the flowers102 dataset with the transforms
        trainData = datasets.Flowers102('../data', split='train', download=True, transform=transform)

        testData = datasets.Flowers102('../data', split='test', download=True, transform=transform)

        learningConfig = config.learning

        train_loader = torch.utils.data.DataLoader(trainData,
                                                batch_size=learningConfig.train_batch_size,
                                                shuffle=True,
                                                drop_last=True)

        test_loader = torch.utils.data.DataLoader(testData,
                                                batch_size=learningConfig.eval_batch_size,
                                                shuffle=True,
                                                drop_last=True)

        return train_loader, test_loader

    elif config.dataset == 'CIFAR100':
        # Define the transforms to apply
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create an instance of the CIFAR100 dataset with the transforms
        trainData = datasets.CIFAR100('../data', train=True, download=True, transform=transform)

        testData = datasets.CIFAR100('../data', train=False, download=True, transform=transform)

        learningConfig = config.learning

        train_loader = torch.utils.data.DataLoader(trainData,
                                                batch_size=learningConfig.train_batch_size,
                                                shuffle=True,
                                                drop_last=True)

        test_loader = torch.utils.data.DataLoader(testData,
                                                batch_size=learningConfig.eval_batch_size,
                                                shuffle=True,
                                                drop_last=True)

        return train_loader, test_loader

    elif config.dataset == 'EuroSAT':
        # Define the transforms to apply
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create an instance of the euroSAT dataset with the transforms
        trainData = datasets.EuroSAT(root='../data', train=True, download=True, transform=transform)

        testData = datasets.EuroSAT(root='../data', train=True, download=True, transform=transform)

        learningConfig = config.learning

        train_loader = torch.utils.data.DataLoader(trainData,
                                                batch_size=learningConfig.train_batch_size,
                                                shuffle=True,
                                                drop_last=True)

        test_loader = torch.utils.data.DataLoader(testData,
                                                batch_size=learningConfig.eval_batch_size,
                                                shuffle=True,
                                                drop_last=True)

        return train_loader, test_loader
