

from numpy.random import RandomState
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Subset


from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

def train(model, device, train_loader, optimizer, epoch, learningConfig, display=True):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        #Only for H2T Phase 1 this is used to get scores
        if model.inScoreCalcPhase: #loss = CEloss + coef * norm(norm(x, ord=r, axis=1), ord=p)`
            coefLambda = learningConfig.group_lrp_regularizer_coef
            regularization_loss = model.group_lasso_regularization()
            loss = F.cross_entropy(output, target) + coefLambda * regularization_loss

        else:
            #MAKE SURE OUTPUT AND TARGET FORMAT WHEN LOSS IS CALCED...SO CONSISTENT SO TRAINING IS DONE PROPERLY
            #MAKE SURE OUTPUT AND TARGET FORMAT WHEN LOSS IS CALCED...SO CONSISTENT SO TRAINING IS DONE PROPERLY
            #MAKE SURE OUTPUT AND TARGET FORMAT WHEN LOSS IS CALCED...SO CONSISTENT SO TRAINING IS DONE PROPERLY
            #MAKE SURE OUTPUT AND TARGET FORMAT WHEN LOSS IS CALCED...SO CONSISTENT SO TRAINING IS DONE PROPERLY
            #MAKE SURE OUTPUT AND TARGET FORMAT WHEN LOSS IS CALCED...SO CONSISTENT SO TRAINING IS DONE PROPERLY
            #MAKE SURE OUTPUT AND TARGET FORMAT WHEN LOSS IS CALCED...SO CONSISTENT SO TRAINING IS DONE PROPERLY

            print('MAKE SURE OUTPUT AND TARGET FORMAT WHEN LOSS IS CALCED...SO CONSISTENT SO TRAINING IS DONE PROPERLY')
            loss = F.cross_entropy(output, target)

        loss.backward()
        optimizer.step()

        #Wandb logging
        wandb.log({"Train Loss Per Batch": loss.item()})
        torch.cuda.empty_cache() # Necessary for efficiency and cuda errors. Maybe even put somewhere in train function for each batch

    if display:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_lossEpoch = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_lossBatch = F.cross_entropy(output, target, size_average=False).item()        
            test_lossEpoch += test_lossBatch # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            #Wandb logging
            wandb.log({"Test Loss Per Batch": test_lossBatch})
            torch.cuda.empty_cache() # Necessary for efficiency and cuda errors. Maybe even put somewhere in train function for each batch

    test_lossEpoch /= len(test_loader.dataset)
    accuracyTest = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_lossEpoch, correct, len(test_loader.dataset),
        accuracyTest))

    #Wandb logging
    wandb.log({"Test Loss Per Epoch": test_lossEpoch, "Test Accuracy Per Epoch": accuracyTest})
    return accuracyTest
