import configs.helper as helper
import configs.trainTestConfig as trainTest
import models.LinearFT as linearFT
import torch.nn as nn
import torch
import numpy as np



def evaluate(datasetName, finetune_backbone=False):
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  print(device) # you will really need gpu's for this part




  accs = []


  # Load the pre-trained ResNet-50 model
  model = models.resnet50(pretrained=True) #########MAKE CLASS FOR MODEL AND INCLDUE THIS THERE

  helper.freezeBackbone(model)

  in_features = model.fc.in_features #The fc layer of resenet50 is Linear(in_features=2048, out_features=1000, bias=True) so storing the 2048 and replacing this to map from 2048 to numClasses for target task ====can see the fc layer like this: backbone = models.resnet50(pretrained=True) => print(backbone.fc)
  targetTaskOutFeatures = helper.numUniqueClasses(datasetName) # num of classes in target task


  #New output head
  classifier = nn.Linear(in_features, targetTaskOutFeatures, bias=True)  # Create a new classifier
  model.fc = classifier  # Replace the classifier layer



  optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

  model.to(device)
  for epoch in range(5):
    trainTest.train(model, device, trainTest.train_loader, optimizer, epoch, display=True)

  accs.append(trainTest.test(model, device, trainTest.test_dataloader))

  accs = np.array(accs)
  print('Acc over 1 instances: %.2f +- %.2f'%(accs.mean(),accs.std()))