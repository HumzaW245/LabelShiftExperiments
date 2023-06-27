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
        
        #New output head
        in_features = self.model.fc.in_features #The fc layer of resenet50 is Linear(in_features=2048, out_features=1000, bias=True) so storing the 2048 and replacing this to map from 2048 to numClasses for target task ====can see the fc layer like this: backbone = models.resnet50(pretrained=True) => print(backbone.fc)
        targetTaskOutFeatures = helper.numUniqueClasses(self.datasetName) # num of classes in target task        
        classifier = nn.Linear(in_features, targetTaskOutFeatures, bias=True)  # Create a new classifier
        classifier.weight.requires_grad = True
        classifier.bias.requires_grad = True
        self.model.fc = classifier  # Replace the classifier layer

        # Get the layers of the model
        
        #self.features = nn.Sequential(*list(self.model.children()))
        #self.layers = list(self.model.children()) #Use .modules() instead of .children() if want to also access layers within layers. (nested layers) since children() looks at layers but there may be a sequential layer among these which contains other layers. to access the nested sequential layer, modules() is more appropriate

    def forward(self, x):
        
        '''
        ##############

        DONT ACTUALLY NEED TO GET EVERY LAYER....
        
        JUST NEED TO GET INDICES OF NEURONS FROM EACH LAYER THAT WILL BE USED.

        ONCE YOU MAKE A CONCATENATED LAYER OF ALL THOSE NEURONS e.g. (l1_n23, l1_n53, l2_n3, ...),
        in the forward pass can continue just doing model(x) and have model.fc = Identity

        so then once done preTrained model(x), it stops, gets the concatenated neurons layer

        and do that layer as the 'x' for a new classifier Linear(neuronsInNewLayer, 10)

        THEN  in backward, it will go from the 10 outputs to the concatenated layer and each neuron from there goes back
        to whatever was used to compute it and updates accordingly in optimizer step.
        ##############


        '''
        # See https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)

        return x


    def _select_features(self, dataset, learning_config):
      return selected_feature_indices, mean_scores 

    def _optimize_finetune(
              self, learning_config, support_dataset, query_dataset,
              selected_feature_indices=None,
              return_output_head=False, customInitHead=None):
        return null
      
