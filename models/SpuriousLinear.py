import torch
import torch.nn as nn
import configs.helper as helper
import torchvision.models as models


class Net(torch.nn.Module):
    def __init__(self, config, n_classes, finetune_backbones):
        super(Net, self).__init__()
        self.numSteps = 0
        
        #False by default since this condition is specific to head2toe
        self.inScoreCalcPhase = False #This is just so evaluate.py, trainTestConfig.py, etc can do model.inScoreCalcPhase. 

        # Dataset
        self.datasetName = config.dataset,

        spuriousConfig = config.spuriousConfig
        
        self.model = models.resnet50(pretrained=spuriousConfig.pretrained_model) 
        
        #New output head
        in_features = self.model.fc.in_features #The fc layer of resenet50 is Linear(in_features=2048, out_features=1000, bias=True) so storing the 2048 and replacing this to map from 2048 to numClasses for target task ====can see the fc layer like this: backbone = models.resnet50(pretrained=True) => print(backbone.fc)
        targetTaskOutFeatures = n_classes
        classifier = nn.Linear(in_features, targetTaskOutFeatures, bias=True)  # Create a new classifier
        classifier.weight.requires_grad = True
        classifier.bias.requires_grad = True
        self.model.fc = classifier  # Replace the classifier layer

        # Load the Checkpoint
        checkpointDirectory = spuriousConfig.checkpointDirectory
        checkpoint = torch.load(checkpointDirectory + '/final_checkpoint.pt')

        # Load the model weights from the checkpoint
        model_state_dict = self.model.state_dict()
        for key in checkpoint.keys():
            if key in model_state_dict:
                model_state_dict[key] = checkpoint[key]

        # Load the modified state dict into the model
        self.model.load_state_dict(model_state_dict)

        # IF WANT TO USE PRETRAINED CHECKPOINT MODEL
        if spuriousConfig.resume is not None:
          print('Resuming from checkpoint at {}...'.format(spuriousConfig.resume))
          checkpoint = torch.load(spuriousConfig.resume)
          model.load_state_dict(checkpoint)
          # Load weights into the modified model, skipping layers that have changed
          self.model.load_state_dict(state_dict, strict=False)


        # FT backbone
        self.finetune_backbones = finetune_backbones
        if self.finetune_backbones == False:
          helper.freezeBackbone(self.model)
          


    def forward(self, x):
        x = self.model(x)
        return x

    def getNumSteps(self):
      return self.numSteps

    def setFinetuneBackbone(self, boolVal):
      self.finetune_backbone = boolVal
      if(self.finetune_backbone == False):
        helper.freezeBackbone(self.model)
      
      self.model.fc.weight.requires_grad = True
      self.model.fc.bias.requires_grad = True

    def setNumSteps(self, newVal):
      self.numSteps = newVal