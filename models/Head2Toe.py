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
        self.model = models.resnet50(pretrained=True)
        
        # FT backbone
        self.finetune_backbones = finetune_backbones
        if self.finetune_backbones == False:
            helper.freezeBackbone(self.model)
        
        # New output head
        in_features = self.model.fc.in_features
        targetTaskOutFeatures = helper.numUniqueClasses(self.datasetName)
        classifier = nn.Linear(in_features, targetTaskOutFeatures, bias=True)
        classifier.weight.requires_grad = True
        classifier.bias.requires_grad = True
        self.model.fc = classifier

        # Get the layers of the model
        self.layers = list(self.model.children()) #Use .modules() instead of .children() if want to also access layers within layers. (nested layers) since children() looks at layers but there may be a sequential layer among these which contains other layers. to access the nested sequential layer, modules() is more appropriate

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


    def _select_features(self, dataset, learning_config):
      return selected_feature_indices, mean_scores 

    def _optimize_finetune(
              self, learning_config, support_dataset, query_dataset,
              selected_feature_indices=None,
              return_output_head=False, customInitHead=None):
        return null
      
