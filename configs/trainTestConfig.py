

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
    countOfDataProcessed = 0
    for batch_idx, batch in enumerate(train_loader):

        '''
        Exiting if number of gradient updates > num_steps configured, even if epochs not finished
        '''
        numStepsDone = model.getNumSteps()
        if numStepsDone > learningConfig.num_steps:
            print(f'Did not update gradient at epoch {epoch} since max number of steps set in configuration is reached: max steps ={learningConfig.num_steps}')
            break
        model.setNumSteps(numStepsDone + 1)


        data = batch[0] 
        countOfDataProcessed += len(data) # For tracking since sometimes len of dataloader caused issues/inaccurate size when dividing
        target = batch[1]
        #group = batch[2] #For spurious datasets (e.g. Waterbirds loader, index 2 and 3 have the group and place for each data point)
        #place = batch[3]
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

            #print('MAKE SURE OUTPUT AND TARGET FORMAT WHEN LOSS IS CALCED...SO CONSISTENT SO TRAINING IS DONE PROPERLY')
            #print(f'output first row is {output[0]} and has type {type(output)} and target first row is {target[0]} and has type {type(target)}  ')
            loss = F.cross_entropy(output, target)

        loss.backward()
        optimizer.step()

        #Wandb logging
        wandb.log({"Train Loss Per Batch": loss.item()})
        
        # If a custom number of steps is used, only track it if not too many things to print. Meant to be used when epochs only print losses for a few epochs since number of steps max limit in config makes gradient updates stop even if num epochs is not reached
        if learningConfig.num_steps < 5000: #5000 is an arbitrary number for the threshold at which printing to console should be done
            print(f'Train step: {numStepsDone} --- Loss = {loss.item()}')
        torch.cuda.empty_cache() # Necessary for efficiency and cuda errors. Maybe even put somewhere in train function for each batch

    if display and numStepsDone <= learningConfig.num_steps:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_lossEpoch = 0
    correct = 0
    countOfDataProcessed = 0
    with torch.no_grad():
        for batch in test_loader:
            data = batch[0] 
            target = batch[1]
            data, target = data.to(device), target.to(device)
            # print(f'Data shape is {data.shape}\
            #         Target shape is {target.shape}\
            #         len(test_loader) = {len(test_loader)}\
            #         current Batch len = {len(batch)}\
            #         length of data is = {len(data)}')
            output = model(data)
            test_lossBatch = F.cross_entropy(output, target, size_average=False).item()        
            test_lossEpoch += test_lossBatch # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            countOfDataProcessed += len(data)
            
            #Wandb logging
            wandb.log({"Test Loss Per Batch": test_lossBatch})
            torch.cuda.empty_cache() # Necessary for efficiency and cuda errors. Maybe even put somewhere in train function for each batch

    test_lossEpoch /= countOfDataProcessed
    accuracyTest = 100. * correct / countOfDataProcessed
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_lossEpoch, correct, countOfDataProcessed,
        accuracyTest))

    #Wandb logging
    wandb.log({"Test Loss Per Epoch": test_lossEpoch, "Test Accuracy Per Epoch": accuracyTest})
    return accuracyTest
