'''



*****************NEED TO first HAVE SPURIOUSLINEAR BERT BASELINE LIKE SPURIOUS CORRELATION PAPER FOR BASELINE ************





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

def _bert_replace_fc(model):
    model.fc = model.classifier
    delattr(model, "classifier")

    def classifier(self, x):
        return self.fc(x)
    
    model.classifier = types.MethodType(classifier, model)

    model.base_forward = model.forward

    def forward(self, x):
        return self.base_forward(
            input_ids=x[:, :, 0],
            attention_mask=x[:, :, 1],
            token_type_ids=x[:, :, 2]).logits

    model.forward = types.MethodType(forward, model)
    return model


def bert_pretrained(output_dim):
	return _bert_replace_fc(BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=output_dim))

class Net(torch.nn.Module):
    def __init__(self, config, n_classes, finetune_backbone, targetSize, concatLayerSize, inScoreCalcPhase, selected_feature_indices, custom_outputHead, custome_preTrainedModel=None):
        super(Net, self).__init__()
        
        # Dataset
        self.datasetName = config.dataset

        spuriousConfig = config.spuriousConfig
        
        self.model = bert_pretrained(n_classes) if custome_preTrainedModel is None else custome_preTrainedModel
        
        self.selected_feature_indices = selected_feature_indices
        self.inScoreCalcPhase = inScoreCalcPhase
        
        # Set the dimensions for BERT's output
        self.targetSize = self.model.config.hidden_size if targetSize is None else targetSize
        self.concatLayerSize = self.model.config.hidden_size if concatLayerSize is None else concatLayerSize

        self.model.fc = nn.Identity()  # Replace the pooler layer with Identity

        self.finetune_backbone = finetune_backbone
        self.helper = helper
        self.helper.FTBackbone(self.model, self.finetune_backbone)

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

    def save_outputs_hook(self, name):
        def hook(module, input, output):
            self._layersChosen[name] = output
        return hook

    def getConcatenatedLayer(self, selected_features):
        # Concatenate features for the classifier
        concatenated_features = torch.cat([selected_features[layer] for layer in self.selected_feature_indices], dim=1)
        return concatenated_features

    def forward(self, x): 
        fwdPassBeforeClassifier = self.model(x)  # Ensure the model processes the input and stores outputs
        selected_features = self._layersChosen 
        
        concatenated_features = self.getConcatenatedLayer(selected_features)
        
        if self.inScoreCalcPhase == False: 
          concatenated_features = concatenated_features[:, self.selected_feature_indices]
        
        x = self.newOutputHead(concatenated_features)  # Pass the concatenated features through the custom classifier
        return x
