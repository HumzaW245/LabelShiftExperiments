
from torchvision import datasets,transforms
import torch
import numpy as np


import torchvision.transforms as transforms
from torchvision.datasets import SVHN

# Define the transforms to apply (--------------------------------------PREPROCESSING FOR EACH DATASET WHAT IS BEST TO GO WITH IMAGENETR50)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create an instance of the SVHN dataset with the transforms
trainData = datasets.SVHN('../data', split='train', download=True, transform=transform)

testData = datasets.SVHN('../data', split='test', download=True, transform=transform)


train_loader = torch.utils.data.DataLoader(trainData,
                                           batch_size=256,
                                           shuffle=True,
                                           drop_last=True)

test_dataloader = torch.utils.data.DataLoader(testData,
                                          batch_size=256,
                                          shuffle=True,
                                          drop_last=True) #Drop last is really just to make sure if last batch is not of equal size, drop it. Nothing to do with setting it aside for testing
