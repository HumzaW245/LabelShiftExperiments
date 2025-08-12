import torch
import torch.nn as nn
import configs.helper as helper
import torchvision.models as models


class Net(torch.nn.Module):
    def __init__(self, config, n_classes, finetune_backbones):
        super(Net, self).__init__()
        
        #False by default since this condition is specific to head2toe
        self.inScoreCalcPhase = False #This is just so evaluate.py, trainTestConfig.py, etc can do model.inScoreCalcPhase. 

        # Dataset
        self.datasetName = config.dataset,

        spuriousConfig = config.spuriousConfig
        
        self.model = helper.bert_pretrained(n_classes) 
        
        #New output head
        in_features = self.model.fc.in_features 
        self.targetTaskOutFeatures = n_classes
        classifier = nn.Linear(in_features, self.targetTaskOutFeatures)  # Create a new classifier
        classifier.weight.requires_grad = True
        classifier.bias.requires_grad = True
        self.model.fc = classifier  # Replace the classifier layer

        # FT backbone
        self.finetune_backbones = finetune_backbones
        helper.FTBackbone(self.model, self.finetune_backbones)
        
          


    def forward(self, x):
        x = self.model(x)
        return x


    def setFinetuneBackbone(self, boolVal):
      self.finetune_backbones = boolVal
      helper.FTBackbone(self.model, self.finetune_backbones)
      
      self.model.fc.weight.requires_grad = True
      self.model.fc.bias.requires_grad = True

    def resetClassificationLayer(self):
      classifier = nn.Linear(self.model.fc.in_features, self.targetTaskOutFeatures)  # Create a new classifier
      classifier.weight.requires_grad = True
      classifier.bias.requires_grad = True
      self.model.fc = classifier
    