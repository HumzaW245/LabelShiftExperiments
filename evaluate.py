import configs.helper as helper
import configs.trainTestConfig as trainTest
import models.LinearFT as linearFT
import models.Head2Toe as h2t
import torch.nn as nn
import torch
import numpy as np
import input_pipeline as pipeLine
import configs.config as config
import wandb

def evaluate(config):
  
  wandb.login()
  #------------------------------------------------------>INITIALIZING WANDB PROJECT NAME AND NAME OF RUN <--------------------------------------------------
  wandb.init(project="Train And Test Accuracy and Losses - Pytorch", name=(config.dataset + ' (' + config.runTypeNameForWandB + ')' ) )

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  print(device) # you will really need gpu's for this part




  accs = []

  learningConfig = config.learning
  # Load the pre-trained ResNet-50 model
  
  if(learningConfig.useH2T): 
    model = h2t.Net(config.dataset, learningConfig.finetune_backbones)
  else:
    model = linearFT.Net(config.dataset, learningConfig.finetune_backbones)



  if learningConfig.optimizer == 'adam':
    optimizer = torch.optim.Adam(
      model.parameters(), 
      lr=learningConfig.learning_rate, 
      weight_decay=learningConfig.weight_decay
      )
  elif learningConfig.optimizer == 'SGD':
    optimizer = torch.optim.SGD(
      model.parameters(), 
      lr=learningConfig.learning_rate, 
      weight_decay=learningConfig.weight_decay, 
      momentum=learningConfig.momentum
      )

  else:
    raise ValueError("the config optimizer used is not supported. Needs to be defined where others are defined like SGD and adam.")

  model.to(device)
  print("\n\n\n")
  print(device)
  print("\n\n\n")


  #Get dataloaders
  train_loader, test_loader = pipeLine.getTrainTestLoaders(config)


  for epoch in range(learningConfig.epochs):
    trainTest.train(model, device, train_loader, optimizer, epoch, display=config.printTraining)

  accs.append(trainTest.test(model, device, test_loader))

  accs = np.array(accs)
  print(f'Accuracy: {accs.mean()}')


'''
Executing Run (optional: with a custom config)
'''

#custom_config = 'learning_rate=0.1, epochs=2, train_batch_size=32, printTraining=True'
config = config.get_config("")

evaluate(config)