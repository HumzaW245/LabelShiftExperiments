import configs.helper as helper
import configs.trainTestConfig as trainTest
import models.LinearFT as linearFT
import models.Head2Toe as h2t
import torch.nn as nn
import torch
import numpy as np
import input_pipeline as pipeLine
import configs.config as config
import sys
#import wandb

def evaluate(config):
  
  #Making sure config dictionary is update to have values expected
  print(f'\n\n\nThe configuration for this run is as follows: \n {config} \n\n\n')


  #wandb.login()
  #------------------------------------------------------>INITIALIZING WANDB PROJECT NAME AND NAME OF RUN <--------------------------------------------------
  #wandb.init(project="Train And Test Accuracy and Losses - Pytorch", name=(config.dataset + ' (' + config.runTypeNameForWandB + ')' ) )

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")


  #to make sure not training on cpu of cluster (login node)
  if device == 'cpu':
    print('\n\n CONNECTED TO A LOGIN NODE SINCE CPU BEING USED --- EXITING --- CONNECT TO A COMPUTE NODE WITH GPU FOR HEAVY COMPUTATION TASKS \n\n')
    sys.exit()

  print(device) # you will really need gpu's for this part


  #Get dataloaders
  train_loader, test_loader = pipeLine.getTrainTestLoaders(config)

  accs = []

  learningConfig = config.learning
  # Load the pre-trained ResNet-50 model
  


  '''
  ----------------------------------------------------------------Head2Toe Base------------------------------------------------------------
  '''
  if(learningConfig.useH2T): 
    
    '''
    =============================Phase 1 to get scores and top scores as indices for phase2 to use as linear head===========================
    '''
    model = h2t.Net(config.dataset, False, learningConfig.target_size, learningConfig.concatLayerSize, True, None, None) #This model is the initialization of phase1 to calculate scores so we don't use finetuning in this step
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

    layersUsedForTopFPctIndicesSelected = helper.layersForTopFPctIndicesSelected(selected_feature_indices, model.getLayersWithRangesOfIndicesAfterProcessing())
    #helper.plotLayersSelectedFeaturesPct(layersUsedForTopFPctIndicesSelected, learningConfig)
    #print(f'\n\n The top {learningConfig.fraction_F * 100} % features selected are as below: \n \n {layersUsedForTopFPctIndicesSelected}')
    print(f'\n\n PHASE 1 COMPLETE --- Selected features have size {selected_feature_indices.shape} and are {selected_feature_indices}')



    '''
    =============================Early Convergence Init: To get weights if want a roughly trained Linear Layer to use as initializor for Phase 2===========================

    Note:Finetune backbones is set to FALSE since we dont want to alter backbone here. Just want a roughly initialized head so in final phase if there is FT, any changes in backbone params is not extreme since initialization won't be random
    **The if condition checks if config is set to check if we have FT = TRUE because otherwise it's just a H2T experiment with no FT used to initialize another H2T experiment with no FT

    '''

    if learningConfig.use_early_conv_phase and learningConfig.finetune_backbones == True: 
      model_early_conv = h2t.Net(config.dataset, False, learningConfig.target_size, newConcatLayerSize, False, selected_feature_indices, None) #FT can be T/F since H2T can be with or without FT
      

      optimizer = helper.getOptimizer(model_early_conv, learningConfig)
      model_early_conv.to(device)
      for epoch in range(learningConfig.early_conv_epochs): #Different epochs for early convergence phase (low since want a roughly trained outputHead)
        trainTest.train(model_early_conv, device, train_loader, optimizer, epoch, learningConfig, display=config.printTraining)
      
      custom_outputHead = model_early_conv.getOutputHead()
    
    else:
      custom_outputHead = None


    '''
    =============================Phase 2: model INITIALIZATION with selected features from phase 1===========================
    '''
    model = h2t.Net(config.dataset, learningConfig.finetune_backbones, learningConfig.target_size, newConcatLayerSize, False, selected_feature_indices, custom_outputHead) #FT can be T/F since H2T can be with or without FT
    print(f'With selected features, this next phase has finetune backbone set to {model.finetune_backbone}') 
 

  # Linear Model
  else:
    print(f"\n\n\nUSING -------------- LINEAR MODEL with FT = {learningConfig.finetune_backbones}-----------------------\n\n\n")
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

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Customize configuration settings.')
    parser.add_argument('--config_string', type=str, help='Comma-separated key-value pairs to update config.')

    args = parser.parse_args()

    custom_config = args.config_string
    config = config.get_config(custom_config)

    #Log File name
    log_file = config.dataset + ' (' + config.runTypeNameForWandB + ').log'
    
    # Open the file in write mode
    with open(log_file, "w") as f:
        # Redirect the standard output to the file
        sys.stdout = f

        # Your code here (all printed outputs will be saved to the log file)
        evaluate(config)

    # Reset the standard output to the console (optional, but recommended)
    sys.stdout = sys.__stdout__
