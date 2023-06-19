import configs.helper as helper
import configs.trainTestConfig as trainTest
import models.LinearFT as linearFT
import torch.nn as nn
import torch
import numpy as np
import input_pipeline as pipeLine


def evaluate(datasetName, finetune_backbone=False):
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  print(device) # you will really need gpu's for this part




  accs = []


  # Load the pre-trained ResNet-50 model
  model = linearFT.Net(datasetName, finetune_backbone)



  optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

  model.to(device)
  print("\n\n\n")
  print(device)
  print("\n\n\n")

  for epoch in range(5):
    trainTest.train(model, device, pipeLine.train_loader, optimizer, epoch, display=True)

  accs.append(trainTest.test(model, device, pipeLine.test_dataloader))

  accs = np.array(accs)
  print('Acc over 1 instances: %.2f +- %.2f'%(accs.mean(),accs.std()))


evaluate('SVHN', finetune_backbone=False)