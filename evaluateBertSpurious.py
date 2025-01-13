import configs.helper as helper
import configs.spuriousTrainTestConfig as trainTest
import models.Bert_SpuriousLinear as bertSpuriousLinear
import models.Bert_SpuriousH2T as bertSpuriousH2T
import torch.nn as nn
import torch
import numpy as np
import input_pipelineSpurious as pipeLine
import configs.config as config
import wandb
import copy

def evaluate(config):
  
  wandb.login()
  #------------------------------------------------------>INITIALIZING WANDB PROJECT NAME AND NAME OF RUN <--------------------------------------------------
  wandb.init(project="Train And Test Accuracy and Losses - Pytorch", name=(config.dataset + ' (' + config.runTypeNameForWandB + ')' ) )

  #Making sure config dictionary is update to have values expected
  print(f'\n\n\nThe configuration for this run is as follows: \n {config} \n\n\n')


  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  print(device)

  #Get dataloaders and n_classes ***ALSO TEST LOADERS ARE RETURNS AS DICT CONTATINING TEST AND VALIDATION LOADERS
  train_loader, train_loader_rw, test_loader_dict, test_loader_dict_rw, n_classes = pipeLine.getTrainTestLoaders(config)
  train_loader2, train_loader_rw2, test_loader_dict2, test_loader_dict_rw2, n_classes = pipeLine.getTrainTestLoaders(config)
  
  #For test and validation, using a separate balanced dataset to get test accuracy after DFR phase
  test_loader = test_loader_dict['wb']
  validation_loader = test_loader_dict['wb_val']

  test_loader_rw = test_loader_dict_rw['wb']
  validation_loader_rw = test_loader_dict_rw['wb_val']

  # Make a deep copy of the dictionary
  validation_loader_rw2 = test_loader_dict_rw2['wb_val']

  


  learningConfig = config.learning
  spuriousConfig = config.spuriousConfig



  '''
  ----------------------------------------------------------------Head2Toe Base------------------------------------------------------------
  '''
  if(learningConfig.useH2T): 
    
    '''
    PHASE 0: SpuriousFeatures with Head2Toe approach first needs to finetune pretrained model on target domain unbalanced dataset
    '''
    custome_preTrainedModel = helper.getBERTModelAfterLinearRun(config, n_classes, learningConfig, device, train_loader, test_loader, finetune_backbone = True)

    '''
    Phase 1: Getting selected_features_indices
    We get scores and top scores as indices for phase2 to use as linear head===========================
    '''
    model = bertSpuriousH2T.Net(config, n_classes, False, learningConfig.target_size, learningConfig.concatLayerSize, True, None, None, custome_preTrainedModel) #This model is the initialization of phase1 to calculate scores so we don't use finetuning in this step
    print(f'To determine scores and select features, this phase has finetune backbone set to {model.finetune_backbone}')

    print(f'setting new optimizer using config.py')
    optimizer = helper.bert_adamw_optimizer(model, learningConfig)  
    
    if learningConfig.scheduler:
      print("USING BERT_LR_SCHEDULER")
      scheduler = helper.bert_lr_scheduler(optimizer, learningConfig.h2tScoreCalcPhaseEpochs)
    model.to(device)
    #Using Validation_loader_rw after workshop paper
    print(f'Length of validation_loader_rw is {len(validation_loader_rw)}')
    for epoch in range(learningConfig.h2tScoreCalcPhaseEpochs):
      trainTest.train(model, device, validation_loader_rw2, optimizer, epoch, learningConfig, display=config.printTraining)
      if learningConfig.scheduler:
        scheduler.step()
    
    outputHeadWeights = model.getOutputHeadLayerWeights()

                                                             
    scores = helper.getScoresAfterTrainingWithGroupLRP(device, outputHeadWeights, setEarlyLayersScoreToZero = learningConfig.setEarlyLayersScoreToZero)
    print(f'This the scores matrix shape after phase 1: {scores.shape}') #Take indices of top F% and pass as indices in 2nd phase
    
    #Initializing another model and using selected_feature_indices
  
    selected_feature_indices = helper.getIndicesOfTopFscores(device, learningConfig.fraction_F, scores)
    newConcatLayerSize = len(selected_feature_indices)
    print(f'New concat layer with selected features will have {newConcatLayerSize} incoming features ')

    #Sparsity check after 
    print("--Using scores----SPARSITY CHECK AFTER SCORES ARE CALCULATED (Scores matrix sparsity right after selecting topFpctScores)-----------")
    helper.printAndPlotSparsityOfSelectedFeaturesIndices(scores[selected_feature_indices], learningConfig, spuriousConfig, "ONLY Selected SCORES Matrix sparsity")
    

    layersUsedForTopFPctIndicesSelected = helper.layersForTopFPctIndicesSelected(selected_feature_indices, model.getLayersWithRangesOfIndicesAfterProcessing())
    helper.plotLayersSelectedFeaturesPct(layersUsedForTopFPctIndicesSelected, learningConfig, "layersUsedForTopFPctIndicesSelected")
    #print(f'\n\n The top {learningConfig.fraction_F * 100} % features selected are as below: \n \n {layersUsedForTopFPctIndicesSelected}')
    print(f'\n\n PHASE 1 COMPLETE --- The top {learningConfig.fraction_F * 100} % features selected have size {selected_feature_indices.shape} and are {selected_feature_indices}')
    
    custom_outputHead = None #This is in case we want a custom initialized classification layer (was initially setup to see if using roughly trained output head to initialize here before training would be effective)


    '''
    =============================Phase 2: model INITIALIZATION with selected features from phase 1===========================
    '''
    # So always setting finetune backbones to False here
    model = bertSpuriousH2T.Net(config, n_classes, False, learningConfig.target_size, newConcatLayerSize, False, selected_feature_indices, custom_outputHead, custome_preTrainedModel) #FT can be T/F since H2T can be with or without FT
    print(f'With selected features, this next phase has finetune backbone set to {model.finetune_backbone}') 



  # Linear Model
  else:
    model = helper.getBERTModelAfterLinearRun(config, n_classes, learningConfig, device, train_loader, test_loader, finetune_backbone = True)


  # DFR Training. 
  if spuriousConfig.reweight_classes or spuriousConfig.reweight_groups or spuriousConfig.reweight_places:
    print('-----------------------------------------------DFR STARTING-------------------------------------------------------')
    
    '''
    Hyperparameter search and setting for DFR phase
    This is essentially doing DFR on half of the validation_loader_rw and then testing it on the other half
    based on different hyperparameters
    '''
    if learningConfig.useH2T:
      helper.setBERTBestHyperparameters(model, device, validation_loader_rw, config, learningConfig, spuriousConfig, display=config.printTraining)
      print('NEW HYPERPARAMETERS CHANGED AFTER TUNING')
      print(learningConfig['DFR_learning_rate'])
      print(learningConfig['DFR_weight_decay'])
      print(learningConfig['DFR_momentum'])
      print(learningConfig['DFRepochs'])
    
    model.setFinetuneBackbone(learningConfig.useFT_DFR_Phase) # DFR Only retraining last layer if not finetuning in dfr phase

    print(f"\n\nFinetune of backbone has been set to {learningConfig.useFT_DFR_Phase}\n\n \
          STARTING DFR Phase with reweighting set for \
          Class = {spuriousConfig.reweight_classes}, \
          Groups = {spuriousConfig.reweight_groups}, \
          Places = {spuriousConfig.reweight_places} \n\n")
    
    '''
    spuriousAffine experiments

    Make Affine parameters of batch norm layers trainable

    '''
    if learningConfig.trainOnlyAffineParamOfBNlayers:  
      print("-------Affine parameters experiment so resetting backbone to not be trainable except for batchNorm layers' affine parameters----------") 
      helper.makeTrainableOnlyAffineParamOfBNlayers(model, learningConfig)
    
    print(f'setting new optimizer using config.py')
    optimizer = helper.bert_adamw_optimizer(model, learningConfig, use_DFR_config=True)  
    
    if learningConfig.scheduler:
      print("USING BERT_LR_SCHEDULER")
      scheduler = helper.bert_lr_scheduler(optimizer, learningConfig.DFRepochs)
    
    model.to(device)      

    for epoch in range(learningConfig.DFRepochs):

      # # SEE Table 2 says use validation data for training(https://arxiv.org/pdf/2204.02937.pdf)
      print(f'DFR phase train about to start with len(val_rw) = {len(validation_loader_rw)}')
      print('DELETE UNNECESSARY PRINTS')
      print('DELETE UNNECESSARY PRINTS')
      print('DELETE UNNECESSARY PRINTS')
      print('change BACK TO VAL_LOADER_RWWWWW')
      print('change BACK TO VAL_LOADER_RWWWWW')
      print('change BACK TO VAL_LOADER_RWWWWW')
      print(f'Length of validation_loader_rw is {len(validation_loader_rw)}')
      trainTest.train(model, device, validation_loader_rw, optimizer, epoch, learningConfig, display=config.printTraining)
      if learningConfig.scheduler:
        scheduler.step()
      
      trainTest.test(model, device, test_loader_rw, learningConfig)

  #Not using DFR so retrain on data that is not reweighted
  else:
    print(f'\n\n-----REWEIGHTING OF DATA IS OFF SO NOW TRAINING With FT = True ON UNBALANCED VALIDATION DATA-----\n\n')
    
    
    model.setFinetuneBackbone(True)

    optimizer = helper.bert_adamw_optimizer(model, learningConfig)

    if learningConfig.scheduler:
      print("USING BERT_LR_SCHEDULER")
      scheduler = helper.bert_lr_scheduler(optimizer, learningConfig.DFRepochs)
    
    model.to(device)    
    for epoch in range(learningConfig.epochs):
      trainTest.train(model, device, validation_loader, optimizer, epoch, learningConfig, display=config.printTraining)
      if learningConfig.scheduler:
              scheduler.step()
      trainTest.test(model, device, test_loader, learningConfig)
    

'''
Executing Run (optional: with a custom config)
'''

#Use by running in command line e.g.: python evaluateSpurious.py --config_string "learning.learning_rate=0.001, learning.epochs=102, learning.train_batch_size=64, learning.finetune_backbones=False, printTraining=False"

import argparse

if __name__ == '__main__':
    print('STARTING PROGRAM')
    
    parser = argparse.ArgumentParser(description='Customize configuration settings.')
    parser.add_argument('--config_string', type=str, help='Comma-separated key-value pairs to update config.')
    parser.add_argument('--path', type=str, required=False, help='Path for dataset use in sh file (SLURM_TMPDIR path)')
    parser.add_argument('--data_path', type=str, required=False, help='Data path argument description (SLURM_TMPDIR path)')
    
    args = parser.parse_args()
    # Print the received arguments for debugging 
    print(f"Config String: {args.config_string}") 
    print(f"Path: {args.path}")
    print(f"Data Path: {args.data_path}")
    custom_config = args.config_string
    config = config.get_config(custom_config)

    evaluate(config)