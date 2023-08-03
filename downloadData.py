'''


Although built-in torch datasets automatically download, we do it here since access to the internet is needed, which is only on the login node.
So we avoid running compute heavy tasks since the login node does not have a gpu and cannot handle heavy computation.

'''

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
import torchvision.models as models


def getData(config):
  
  #Making sure config dictionary is update to have values expected
  print(f'\n\n\nThe configuration for this run is as follows: \n {config} \n\n\n')


  #wandb.login()
  #------------------------------------------------------>INITIALIZING WANDB PROJECT NAME AND NAME OF RUN <--------------------------------------------------
  #wandb.init(project="Train And Test Accuracy and Losses - Pytorch", name=(config.dataset + ' (' + config.runTypeNameForWandB + ')' ) )

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  print(device) # you will really need gpu's for this part


  #Get dataloaders
  train_loader, test_loader = pipeLine.getTrainTestLoaders(config)

  print(f'Dataset : {config.dataset} was downloaded successfully. Can now use it offline.')




def getBackboneDownloaded():
    model = models.resnet50(pretrained=True)

    print(f'\n\nRan the backbone successfully so it can be downloaded locally and is available for offline use \n\n')


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

    getBackboneDownloaded()
    getData(config) #Specify the dataset to download when running this script