import torch
import torch.nn as nn
import configs.helper as helper
import torchvision.models as models




'''

# Setting up code from spurious papaer to work wtih h2t:

0) run the actual cloned repo for DFR (e.g. python3 train_classifier.py) so it downloads the data. Then can easily see where it downloads it and how to use it  etc (https://github.com/PolinaKirichenko/deep_feature_reweighting)
1) Focus on loading data, 
2) USING BASIC LINEAR model ***NO PRETRAINING, SET TO FALSE***, training on unbalanced data, training on balanced data to retrain output layer
3) see if results reflect expected performance mentioned in paper (which versions perform better relative to each other)
4) Make separate class and use existing linear based one to code along the h2t one 
5) move linear version to SpuriousLinear.py file

'''
class Net(torch.nn.Module):
    def __init__(self, config, n_classes, finetune_backbones):
        super(Net, self).__init__()
        
        # Dataset
        self.datasetName = config.dataset,

        # Load the Checkpoint
        spuriousConfig = config.spuriousConfig
        checkpointDirectory = spuriousConfig.checkpointDirectory
        checkpoint = torch.load(checkpointDirectory + '/final_checkpoint.pt')
        state_dict = checkpoint['state_dict']
        original_class_count = checkpoint['class_count']
        self.model = models.resnet50(pretrained=spuriousConfig.pretrained_model) 
        
        # IF WANT TO USE PRETRAINED MODEL
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
          

        #New output head
        in_features = self.model.fc.in_features #The fc layer of resenet50 is Linear(in_features=2048, out_features=1000, bias=True) so storing the 2048 and replacing this to map from 2048 to numClasses for target task ====can see the fc layer like this: backbone = models.resnet50(pretrained=True) => print(backbone.fc)
        targetTaskOutFeatures = n_classes
        classifier = nn.Linear(in_features, targetTaskOutFeatures, bias=True)  # Create a new classifier
        classifier.weight.requires_grad = True
        classifier.bias.requires_grad = True
        self.model.fc = classifier  # Replace the classifier layer

    def forward(self, x):
        x = self.model(x)
        return x