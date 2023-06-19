import torch
import torch.nn as nn
import configs.helper as helper
import torchvision.models as models

class Net(torch.nn.Module):
    def __init__(self, datasetName, finetune_backbones):
        super(Net, self).__init__()
        
        # Dataset
        self.datasetName = datasetName
        # Load the pre-trained ResNet-50 model
        self.model = models.resnet50(pretrained=True) #########MAKE CLASS FOR MODEL AND INCLDUE THIS THERE
        

        # FT backbone
        self.finetune_backbones = finetune_backbones
        if self.finetune_backbones == False:
          helper.freezeBackbone(self.model)
          

        #New output head
        in_features = self.model.fc.in_features #The fc layer of resenet50 is Linear(in_features=2048, out_features=1000, bias=True) so storing the 2048 and replacing this to map from 2048 to numClasses for target task ====can see the fc layer like this: backbone = models.resnet50(pretrained=True) => print(backbone.fc)
        targetTaskOutFeatures = helper.numUniqueClasses(self.datasetName) # num of classes in target task        
        classifier = nn.Linear(in_features, targetTaskOutFeatures, bias=True)  # Create a new classifier
        classifier.weight.requires_grad = True
        classifier.bias.requires_grad = True
        self.model.fc = classifier  # Replace the classifier layer

    def forward(self, x):
        x = self.model(x)
        return x