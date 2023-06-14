import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, device, train_loader, optimizer, lossFunction, epoch, display=True):
    #Set model to training mode ----------------------------------CARE WHEN NORMALIZATION, DIFFERENT FOR TRAIN/TEST---------------------
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        #Data to device
        data, target = data.to(device), target.to(device)
        
        #Reset grad for next batch
        optimizer.zero_grad()
        output = model(data)

        #Loss
        loss = lossFunction(output, target) #F.cross_entropy(output, target)
        
        #Backprop
        loss.backward()

        #Optimization step (Parameter Update)
        optimizer.step()

    #Log results of each epoch. CARE: Epoch will be an integer passed so the parameter is ONLY for tracking here. train(...) needs to be called in a loop based on how many epochs are to be trained (each epoch's index passed as epoch)
    if display:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))
      

def test(model, device, test_loader, lossFunction):
    #Set model to test mode ----------------------------------CARE WHEN NORMALIZATION, DIFFERENT FOR TRAIN/TEST---------------------
    model.eval()

    
    test_loss = 0
    correct = 0
    
    #no_grad() will not hold grad, which is fine since we do not want to calculate gradient as we are only interested in checking the prediction from the forward pass.
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            #Test Loss
            test_loss += lossFunction(output, target, size_average=False).item() #F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
            
            #Predicted class
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            
            #Number of correct predictions in the batch to update overall total of correct predictions
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    #Average test loss over all dataset
    test_loss /= len(test_loader.dataset)

    #Log results
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    #Return Accuracy Percent
    return 100. * correct / len(test_loader.dataset)

