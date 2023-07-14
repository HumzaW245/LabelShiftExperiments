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


  #Get dataloaders
  train_loader, test_loader = pipeLine.getTrainTestLoaders(config)

  accs = []

  learningConfig = config.learning
  # Load the pre-trained ResNet-50 model
  
  if(learningConfig.useH2T): #Phase 1 to get scores and top scores as indices for phase2 to use as linear head
    model = h2t.Net(config.dataset, False, learningConfig.target_size, learningConfig.concatLayerSize, True, None) #This model is the initialization of phase1 to calculate scores so we don't use finetuning in this step
    print(f'To determine scores and select features, this phase has finetune backbone set to {model.finetune_backbone}')

    optimizer = helper.getOptimizer(model, learningConfig)
    model.to(device)
    for epoch in range(learningConfig.epochs):
      trainTest.train(model, device, train_loader, optimizer, epoch, learningConfig, display=config.printTraining)
    
    outputHeadWeights = model.getOutputHeadLayerWeights()
    scores = helper.getScoresAfterTrainingWithGroupLRP(device, outputHeadWeights)
    print(f'This the scores matrix shape after phase 1: {scores.shape}') #Take indices of top F% and pass as indices in 2nd phase
    
    #Initializing another model and using selected_feature_indices
    selected_feature_indices = helper.getIndicesOfTopFscores(device, learningConfig.fraction_F, scores)
    newConcatLayerSize = len(selected_feature_indices)
    print(f'New concat layer with selected features will have {newConcatLayerSize} incoming features ')

    #print(model.getLayersWithRangesOfIndicesAfterProcessing())
    #print(model.getLayersWithRangesOfIndicesAfterProcessing())
    layersUsedForTopFPctIndicesSelected = helper.layersForTopFPctIndicesSelected(selected_feature_indices, model.getLayersWithRangesOfIndicesAfterProcessing())
    helper.plotLayersSelectedFeaturesPct(layersUsedForTopFPctIndicesSelected)
    #print(f'\n\n The top {learningConfig.fraction_F * 100} % features selected are as below: \n \n {layersUsedForTopFPctIndicesSelected}')
    print(f'\n\n PHASE 1 COMPLETE --- Selected features have size {selected_feature_indices.shape} and are {selected_feature_indices}')



    model = h2t.Net(config.dataset, learningConfig.finetune_backbones, learningConfig.target_size, newConcatLayerSize, False, selected_feature_indices) #FT can be T/F since H2T can be with or without FT
    print(f'With selected features, this next phase has finetune backbone set to {model.finetune_backbone}') 
    
  else:
    model = linearFT.Net(config.dataset, learningConfig.finetune_backbones)    
  
  #This will either be the 2nd model with select features optimizer OR if it is a Linear/FT run, it will be the optimizer for that. Cannot put this inside the else condition above
  optimizer = helper.getOptimizer(model, learningConfig)  
  model.to(device)
  print(f"\n\n\n {device} \n\n\n")

  for epoch in range(learningConfig.epochs):
    trainTest.train(model, device, train_loader, optimizer, epoch, learningConfig, display=config.printTraining)

    accs.append(trainTest.test(model, device, test_loader))

  accs = np.array(accs)
  print(f'Accuracy: {accs.mean()}')


'''
Executing Run (optional: with a custom config)
'''

#custom_config = 'learning_rate=0.1, epochs=2, train_batch_size=32, printTraining=True'
config = config.get_config("")

evaluate(config)