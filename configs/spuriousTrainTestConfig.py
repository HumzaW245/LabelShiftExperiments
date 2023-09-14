

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
from collections import defaultdict
import datetime
from collections import Counter
'''

Some aspects like getting worst group accuracy are needed which are not used for non spurious related experiments

which is why this separate trainTest file is created
'''



def train(model, device, train_loader, optimizer, epoch, learningConfig, display=True):

    model.train()
    countOfDataProcessed = 0
    timeAtStartOfEpoch = datetime.datetime.now()
    #Time before loop
    print(f'\n TRAINING EPOCH # {epoch}')
    print("\n\n\n")
    #assert False
    for batch_idx, batch in enumerate(train_loader):
        #print(f'\n ITERATION NUMBER: {batch_idx}')
        #print(f'Time at start of iteration, INSIDE train_loader for loop by batch: {datetime.datetime.now()}')
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
        #------------------UNCOMMENT BELOW TO PRINT GROUP COUNTS FOR EACH BATCH DURING TRAINING-------------------
        # group = batch[2] #For spurious datasets (e.g. Waterbirds loader, index 2 and 3 have the group and place for each data point)
        
        # group_counts = Counter(group.tolist())
        # # Print the counts for each group in the batch
        # print("Group Counts in Batch:")
        # for group_value, count in group_counts.items():
        #     print(f"Group {group_value}: {count}")
        #------------------UNCOMMENT ABOVE TO PRINT GROUP COUNTS FOR EACH BATCH DURING TRAINING-------------------
            
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
        if learningConfig.num_steps < 5000 and numStepsDone % 10 == 0: #5000 is an arbitrary number for the threshold at which printing to console should be done
            print(f'Train step: {numStepsDone} --- Loss = {loss.item()}')
        #torch.cuda.empty_cache() # Necessary for efficiency and cuda errors. Maybe even put somewhere in train function for each batch

    trainTime = datetime.datetime.now() - timeAtStartOfEpoch
    if display and numStepsDone <= learningConfig.num_steps:
      print(f'Train Epoch: {epoch}\tLoss: {loss.item()} \t- Training time for epoch: {trainTime}')


def test(model, device, test_loader):
    model.eval()
    test_lossEpoch = 0
    correct = 0
    countOfDataProcessed = 0
    group_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    startTime = datetime.datetime.now()
    with torch.no_grad():
        for batch in test_loader:
            data = batch[0]
            target = batch[1]
            group = batch[2]
            place = batch[3]
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            test_lossBatch = F.cross_entropy(output, target, size_average=False).item()
            test_lossEpoch += test_lossBatch
            pred = output.max(1, keepdim=True)[1]
            
            # Update group-specific accuracy
            for i in range(len(group)):
                group_id = group[i].item()
                correct += pred[i].eq(target[i].view_as(pred[i])).sum().item()
                group_accuracy[group_id]['correct'] += pred[i].eq(target[i].view_as(pred[i])).sum().item()
                group_accuracy[group_id]['total'] += 1

            countOfDataProcessed += len(data)
            
            wandb.log({"Test Loss Per Batch": test_lossBatch})
            #torch.cuda.empty_cache()

    test_lossEpoch /= countOfDataProcessed
    accuracyTest = 100. * correct / countOfDataProcessed
    
    testTime = datetime.datetime.now() - startTime
    print(f'\nTest set: : Average loss: {test_lossEpoch}\tAccuracy: {correct}/{countOfDataProcessed} ({accuracyTest}%)\t- Test time: {testTime}')

    wandb.log({"Test Loss Per Epoch": test_lossEpoch, "Test Accuracy Per Epoch": accuracyTest})

    # Log group-specific accuracies
    for group_id, group_data in group_accuracy.items():
        group_correct = group_data['correct']
        group_total = group_data['total']
        group_accuracy_percentage = 100. * group_correct / group_total
        print(f'Group {group_id}: Accuracy: {group_correct}/{group_total} ({group_accuracy_percentage:.2f}%)')
        wandb.log({f"Test Accuracy for group {group_id} Per Epoch": group_accuracy_percentage})

    return accuracyTest
