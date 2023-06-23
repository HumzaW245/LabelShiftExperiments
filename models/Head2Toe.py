import torch
import torch.nn as nn
import configs.helper as helper
import torchvision.models as models


'''

Refer to https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html for what each layer is doing. 



For example, the flatten is not a layer so have to do it ourselves at layer 9 in forward pass. Search "x = torch.flatten(x, 1)" in the link

'''
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
        
        #self.features = nn.Sequential(*list(self.model.children()))
        #self.layers = list(self.model.children()) #Use .modules() instead of .children() if want to also access layers within layers. (nested layers) since children() looks at layers but there may be a sequential layer among these which contains other layers. to access the nested sequential layer, modules() is more appropriate

    def forward(self, x):
        for idx, module in enumerate(self.model.children()):
            if idx ==  9: #The flatten is not a layer so have to do it ourselves here. Search "x = torch.flatten(x, 1)" in https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html
                #print(f"Reached Layer {idx} so flattening tensor")
                # Flatten the output tensor
                x = torch.flatten(x, 1)
            
            x = module(x)
            #print(f"Layer #: {idx} and module name is {module}")
            #print(f"Layer output shape: {x.shape}")
        return x


    def _select_features(self, dataset, learning_config):
      return selected_feature_indices, mean_scores 

    def _optimize_finetune(
              self, learning_config, support_dataset, query_dataset,
              selected_feature_indices=None,
              return_output_head=False, customInitHead=None):
        return null
      
