import configs.helper as helper
import configs.trainTestConfig as trainTest
import models.SpuriousLinear as spuriousLinear
import models.SpuriousH2T as spuriousH2T
import torch.nn as nn
import torch
import numpy as np
import input_pipelineSpurious as pipeLine
import configs.config as config
import wandb

def evaluate(config):
  
  #Making sure config dictionary is update to have values expected
  print(f'\n\n\nThe configuration for this run is as follows: \n {config} \n\n\n')


  #wandb.login()
  #------------------------------------------------------>INITIALIZING WANDB PROJECT NAME AND NAME OF RUN <--------------------------------------------------
  #wandb.init(project="Train And Test Accuracy and Losses - Pytorch", name=(config.dataset + ' (' + config.runTypeNameForWandB + ')' ) )

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  print(device) # you will really need gpu's for this part


  #Get dataloaders and n_classes ***ALSO TEST LOADERS ARE RETURNS AS DICT CONTATINING TEST AND VALIDATION LOADERS
  train_loader, test_loader_dict, n_classes = pipeLine.getTrainTestLoaders(config)
  test_loader = test_loader_dict['wb']
  validation_loader = test_loader_dict['wb_val']
  
  accs = []

  learningConfig = config.learning
  spuriousConfig = config.spuriousConfig

  # Linear Model
  print(f"\n\n\nUSING -------------- LINEAR MODEL with FT = {learningConfig.finetune_backbones}-----------------------\n\n\n")
  model = spuriousH2T.Net(config, n_classes, learningConfig.finetune_backbones)    
  
  #This will either be the 2nd model with select features optimizer OR if it is a Linear/FT run, it will be the optimizer for that. Cannot put this inside the else condition above
  
##################################################TO DO:########################################################################################
##################################################TO DO:########################################################################################
##################################################TO DO:########################################################################################
##################################################TO DO:########################################################################################
##################################################TO DO:########################################################################################  
#############NEED TO USE VALUES GIVEN IN SPURIOUS CONFIG FOR OPTIMIZER HERE. 
#############TO FIX!! MAYBE JUST ADD THE SAME NAME AND PARAMS WITH DEFAULTS IN BOTH SO CAN RUN WITH BOTH LEARNING AND SPURIOUS CONFIG
  print(f'getting optimizer')
  optimizer = helper.getOptimizer(model, spuriousConfig)  
  model.to(device)
  print(f"\n\n\n {device} \n\n\n")

  for epoch in range(learningConfig.epochs):
    print(f'epoch is {epoch}')
    ###########################CHECKING DATA IS COMING FROM LOADERS PROPERLY AND BALANCED WHEN NEEDED, ETC...
    ###########################CHECKING DATA IS COMING FROM LOADERS PROPERLY AND BALANCED WHEN NEEDED, ETC...
    ###########################CHECKING DATA IS COMING FROM LOADERS PROPERLY AND BALANCED WHEN NEEDED, ETC...
    ###########################CHECKING DATA IS COMING FROM LOADERS PROPERLY AND BALANCED WHEN NEEDED, ETC...
    ###########################CHECKING DATA IS COMING FROM LOADERS PROPERLY AND BALANCED WHEN NEEDED, ETC...
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        print(target)
    ###########################DELETE ABOVE AND UNCOMMENT BELOW trainTest.train function....#########################
    ###########################DELETE ABOVE AND UNCOMMENT BELOW trainTest.train function....#########################
    ###########################DELETE ABOVE AND UNCOMMENT BELOW trainTest.train function....#########################
    ###########################DELETE ABOVE AND UNCOMMENT BELOW trainTest.train function....#########################
    ###########################DELETE ABOVE AND UNCOMMENT BELOW trainTest.train function....#########################
    print(f"UNCOMMENT LINES BELOW IN CODE AFTER DONE VERIFYING LOADERS WORKING PROPERLY\n\n\n\
            MAKE SURE HAVE SETUP FOR VALIDATION STUFF TOOO SINCE VALIDATION SET TOO to be used")
    
    #trainTest.train(model, device, train_loader, optimizer, epoch, learningConfig, display=config.printTraining)

    #accs.append(trainTest.test(model, device, test_loader))

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

    evaluate(config)