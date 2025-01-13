'''

see https://github.com/izmailovpavel/spurious_feature_learning/blob/main/models/text_models.py

def bert_pretrained(output_dim):
	return _bert_replace_fc(BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=output_dim))

then see how still using classifier

https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L644

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

then see how https://github.com/izmailovpavel/spurious_feature_learning/blob/main/dfr_evaluate_spurious.py

still set mode.fc to Identity so can replace it the way class BertForSequenceClassification(BertPreTrainedModel):
does it. by doing self.classifier = nn.Linear(config.hidden_size, config.num_labels)


*****************SEE 'def forward()' for BertForSequenceClassification class,
    can access hidden states 


'''


'''
-------------------Delete all above this line once done=====================
'''

import torch
import torch.nn as nn
import configs.helper as helper
from transformers import BertModel, BertConfig
from transformers import BertForSequenceClassification

from typing import Dict, Iterable, Callable
from torch import Tensor
import math

class Net(torch.nn.Module):
    def __init__(self, config, n_classes, finetune_backbone, targetSize, concatLayerSize, inScoreCalcPhase, selected_feature_indices, custom_outputHead, custome_preTrainedModel=None):
        super(Net, self).__init__()
        
        # Dataset
        self.datasetName = config.dataset

        spuriousConfig = config.spuriousConfig
        
        self.model = helper.bert_pretrained(n_classes) if custome_preTrainedModel is None else custome_preTrainedModel
        
        self.selected_feature_indices = selected_feature_indices
        self.inScoreCalcPhase = inScoreCalcPhase
        
        # Set the dimensions for BERT's output
        self.targetSize = self.model.config.hidden_size if targetSize is None else targetSize
        self.concatLayerSize = self.model.config.hidden_size if concatLayerSize is None else concatLayerSize

        self.model.fc = nn.Identity()  # Replace the pooler layer with Identity

        self.finetune_backbone = finetune_backbone
        helper.FTBackbone(self.model, self.finetune_backbone)

        self.targetTaskOutFeatures = n_classes
      
        if self.inScoreCalcPhase == False: 
          self.newOutputHead = nn.Linear(len(self.selected_feature_indices), self.targetTaskOutFeatures, bias=True)
        else:
          self.newOutputHead = nn.Linear(self.concatLayerSize, self.targetTaskOutFeatures, bias=True) 

        if custom_outputHead is not None:
          with torch.no_grad():
            self.newOutputHead.weight.copy_(custom_outputHead.weight.data)
            self.newOutputHead.bias.copy_(custom_outputHead.bias.data)

        self._layersChosen = {}
        
        self.layersWithRangesOfIndicesAfterProcessing = {} 
        
        for name, module in self.model.named_modules():
          module.register_forward_hook(self.save_outputs_hook(name)) 

    def forward(self, x): 
        fwdPassBeforeClassifier = self.model(x)  # Ensure the model processes the input and stores outputs
        selected_features = self._layersChosen 
        
        concatenated_features = self.getConcatenatedLayer(selected_features)
        
        if self.inScoreCalcPhase == False: 
          concatenated_features = concatenated_features[:, self.selected_feature_indices]
        
        x = self.newOutputHead(concatenated_features)  # Pass the concatenated features through the custom classifier
        return x
    '''
    Forward hooks (Basically telling upon initialization to keep track of modules specified with module.register_forward_hook) - The values being tracked are updated as forward passes are done and since we append those 'variables' to self._layersChosen, can access it always. (So after self.model(x), their values are updated and so self._layersChosen will have the updated values which we can use to build the new concatenated layer)
    
    ***********CARE: Make sure:
          IF using a LIST to append outputs: Clear the list each time forward pass is called so not just appending same features over and over again. AND CHANGE  getConcatenatedLayer() function to work with list

          OR

          Use a dictionary and update value at specific keys.AND CHANGE  getConcatenatedLayer() function to work with dict

    '''
    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(module, input, output):
            #print(f'layer name is {layer_id}')
            #print(f'layer output shape is {output.shape}')
            self._layersChosen[layer_id] = output            # Can use this if want to store the name passed in a dictionary with key = name value = output
        return fn


    '''
    Apply pooling to each layer (outputs of layers) (2D strided for 4 dim shapes, 1D strided for 3 dim shapes, no pooling for 2 dim shapes)
    Flatten
    Normalize each output
    Finally -> concatenate along dim 1 all outputs to make concatenated layer
    '''
    def getConcatenatedLayer(self, selected_features):
      pool_size = 0
      target_size = self.targetSize

      all_features = []

      startRange = 0
      endRange = 0
      for key, output in selected_features.items():

        #2-D Strided pooling when shape is [batch_size, channels, height, width]
        if len(output.shape) == 4: 
          _, channels, height, width = output.shape #Channels first in pytorch

          if channels >= target_size:
            # Global pool.
            pool_size = 0
          else:
            # Assuming square image.
            n_patches_per_row = int(math.sqrt(target_size // channels))
            pool_size = width // n_patches_per_row
          
          if pool_size > 0:
              output = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)(output)
              output = output.flatten(start_dim=1)
          else:
              # Global pool
              output = torch.mean(output, dim=[2, 3]) #dim 2, 3 here since tf uses channels last. Basically want dimension of features which is height and width dimensions ->dims 2, 3 -> [batch_size, channels, height, width]
              

          all_features.append(output)


        #1-D Strided pooling when shape is [batch_size, channels, channelFeatures]
        elif len(output.shape) == 3: 
          _, channels, n_features = output.shape #Channels first in pytorch

          if channels >= target_size:
              # Global pool
              pool_size = 0
          else:
              # Assuming square image
              n_groups = target_size / channels
              pool_size = int(n_features / n_groups)

          if pool_size > 0:
              output = nn.AvgPool1d(kernel_size=pool_size, stride=pool_size)(output)
              output = output.flatten(start_dim=1)
          else:
              # Global pool
              output = torch.mean(output, dim=[2]) #dim 2 here since tf uses channels last. Basically want dimension of features which is at dim = 2 [batch_size, channels, channelFeatures]
          
          all_features.append(output)

        #No pooling when shape is [batch_size, features]
        elif len(output.shape) == 2: 
          all_features.append(output)

        else:
          raise ValueError(
              f'Output tensor: {key} with shape {output.shape} not 2D or 4D.')
        

        #Getting the range for indices belonging to each kind of layer so that we can track how many indices per range are used
        numFeaturesAfterFlattening = output.shape[-1] #Last dimension will be the flattened features dimension. First dim is batch_size
        endRange = endRange + numFeaturesAfterFlattening
        self.layersWithRangesOfIndicesAfterProcessing[str(key)] = (startRange, endRange) # range for key
        startRange = startRange + numFeaturesAfterFlattening #This is after updating keys so next iteration is correct

      '''
      The flatten_and_concat function says its supposed to summarize into a single feature vector
      but it returns a list of tensors instead. 

      Based on my understanding of the paper, we want to train a head with weights Wall so concatenating along features dimension to
      make a layer which will essentially have weights Wall and can be trained with regularization.
      
      Regularization will bring some features' values down and others stay high which gives us a ranking of most important (impactful) features
      and then using a fraction, we can take the top F% of the features based on scores


      *We also normalize each tensor in the list of features since the paper suggested it after flattening.

      '''
      #bef = [tensor.shape for tensor in all_features]
      #print(f'shapes before concat features BEFORE NORMALIZATION{bef}')
      all_features = [torch.nn.functional.normalize(tensor, p=2, dim=1) for tensor in all_features]

      #bef = [tensor.shape for tensor in all_features]
      #print(f'shapes before concat features {bef}')
      
      #concatenating into one tensor
      concatenatedFeatures = torch.cat(all_features, dim=1) #Concatenating flattened layers 




      #print(f'shape of concatenated layer {concatenatedFeatures.shape}')
      return concatenatedFeatures          
      
    def group_lasso_regularization(self): #regularizer_loss = norm(norm(x, ord=r, axis=1), ord=p)`
      w_all = self.getOutputHeadLayerWeights() #Shape is [out_features, in_features] so first norm over out_features
      score_i = torch.norm(w_all, p=2, dim=0) #score_i is basically a l2 norm (p=2 means l2 norm)
      #print(f'score_i is feature i score and shape is: {score_i.shape}')
      regularization_loss = torch.norm(score_i, p=1) #p=1 means l1 norm
      return regularization_loss

    def setToScoreCalcPhase(self, boolVal):
      self.inScoreCalcPhase = boolVal
      
    def setFinetuneBackbone(self, boolVal):
      self.finetune_backbone = boolVal
      helper.FTBackbone(self.model, self.finetune_backbone)

      
      #There is no  model.fc (set to Identity()) for h2t since output head Linear layer is separate from self.model which contains only pretrained model (backbone only)
      #So no need for below requires grad part
      #self.model.fc.weight.requires_grad = True
      #self.model.fc.bias.requires_grad = True

    def getOutputHeadLayerWeights(self):
      return self.newOutputHead.weight #Don't use .weight.data because .weight will track operations in computation graph which is important when doing backprop with regularization loss
      
    def getOutputHead(self):
      return self.newOutputHead
      
    def getLayersWithRangesOfIndicesAfterProcessing(self):
      return self.layersWithRangesOfIndicesAfterProcessing

    def resetClassificationLayer(self):
       self.newOutputHead = nn.Linear(len(self.selected_feature_indices), self.targetTaskOutFeatures, bias=True)  

    def getNumSteps(self):
      return self.numSteps

    def setNumSteps(self, newVal):
      self.numSteps = newVal
