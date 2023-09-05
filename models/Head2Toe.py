import torch
import torch.nn as nn
import configs.helper as helper
import torchvision.models as models

'''
See example 2: https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904

for reference on using forward hooks to get intermediate outputs

'''

from typing import Dict, Iterable, Callable
from torch import Tensor
import math
class Net(torch.nn.Module):
    def __init__(self, datasetName, finetune_backbone, targetSize, concatLayerSize, inScoreCalcPhase, selected_feature_indices, custom_outputHead):
        super(Net, self).__init__()
        self.numSteps = 0
        self.model = models.resnet50(pretrained=True)

        self.selected_feature_indices = selected_feature_indices
        self.inScoreCalcPhase = inScoreCalcPhase

        #in_features = self.model.fc.in_features #The fc layer of resenet50 is Linear(in_features=2048, out_features=1000, bias=True) so storing the 2048 and replacing this to map from 2048 to numClasses for target task ====can see the fc layer like this: backbone = models.resnet50(pretrained=True) => print(backbone.fc)
        
        #Target size is use for pooling of each layer, NOT the final "in_features" value of the Linear layer (output head)
        self.targetSize = targetSize #See comment above...2048 for now used since testing with fc layer as concatenated layer (The fc layer of resenet50 is Linear(in_features=2048, out_features=1000, bias=True))

        
        #Size of output head
        self.concatLayerSize = concatLayerSize #See comment above...2048 for now used since testing with fc layer as concatenated layer (The fc layer of resenet50 is Linear(in_features=2048, out_features=1000, bias=True))

        self.model.fc = nn.Identity()  # Replace the classifier layer with Identity since classifier will be separately applied after features chosen are extracted (See forward function)

        self.finetune_backbone = finetune_backbone
        if(self.finetune_backbone == False):
          helper.freezeBackbone(self.model)

        # Apply adaptive pooling to resize the tensor
        self.adaptive_pool1D = nn.AdaptiveAvgPool1d(self.targetSize)
        self.adaptive_pool2D = nn.AdaptiveAvgPool2d(self.targetSize)

        #New output head
        targetTaskOutFeatures = helper.numUniqueClasses(datasetName) # num of classes in target task
      
        if self.inScoreCalcPhase == False: #so if in 2nd phase, incoming features is # selected_indices
          self.newOutputHead = nn.Linear(len(self.selected_feature_indices), targetTaskOutFeatures, bias=True)  # Create a new classifier
        else:
          self.newOutputHead = nn.Linear(self.concatLayerSize, targetTaskOutFeatures, bias=True)  # Create a new classifier

        if custom_outputHead != None:
          #print(f'Passed custom_output head has weight {custom_outputHead.weight.data}')
          with torch.no_grad():
            self.newOutputHead.weight.copy_(custom_outputHead.weight.data)
            self.newOutputHead.bias.copy_(custom_outputHead.bias.data)

        #Forward hook setup to store intermediate outputs of chosen layers/features
        self._layersChosen = {}
        
        self.layersWithRangesOfIndicesAfterProcessing = {} #This is to track which layer has which indices after flattening outputs

        for name, module in self.model.named_modules():
          module.register_forward_hook(self.save_outputs_hook(name)) #Name is what the layer_id is in save_outputs_hook...usually forward hooks are not callable so just have (Self, module, input, output) but here a function with those is defined so it can call with passed argument

    def forward(self, x): 
        fwdPassBeforeClassifier = self.model(x)
        selected_features = self._layersChosen # At this point, have not gone through classifier but have all chosen features so can now pass this through a linear layer for classification (FIRST NEED TO CONCAT etc and make it passable to linear layers)
        
        #Concatenated Layer for classifier
        concatenated_features = self.getConcatenatedLayer(selected_features) #This is flattening everything passed starting from dim 1 (see definition)
        
        if self.inScoreCalcPhase == False: #So if in 2nd phase and have indices
          #print(f'shape in 2nd phase before selecting features {concatenated_features.shape}')
          #print(f'selected feature indices has size {self.selected_feature_indices.shape} and are {self.selected_feature_indices}')
          concatenated_features = concatenated_features[:, self.selected_feature_indices]
          #print(f'shape in 2nd phase AFTER selecting features {concatenated_features.shape}')
        
        #print(f'shape of concatenated layer BEFORE PASSING THROUGH OUTPUT HEAD {concatenated_features.shape}')
        x = self.newOutputHead(concatenated_features)
        ######################################################
        ######################################################
        # WILL HAVE A NEW 'CONCAT LAYER SIZE' TO ENTER IN CONFIG SINCE ONLY USED IN 2ND PHASE AND SUBSET OF FEATURES USED
        ######################################################
        ######################################################

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
      if(self.finetune_backbone == False):
        helper.freezeBackbone(self.model)
      
      self.model.fc.weight.requires_grad = True
      self.model.fc.bias.requires_grad = True

    def getOutputHeadLayerWeights(self):
      return self.newOutputHead.weight #Don't use .weight.data because .weight will track operations in computation graph which is important when doing backprop with regularization loss
      
    def getOutputHead(self):
      return self.newOutputHead
      
    def getLayersWithRangesOfIndicesAfterProcessing(self):
      return self.layersWithRangesOfIndicesAfterProcessing

    def getNumSteps(self):
      return self.numSteps

    def setNumSteps(self, newVal):
      self.numSteps = newVal