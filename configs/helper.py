import torch

def numUniqueClasses(datasetName):

  datasetsClasses = {'SVHN': 10}
  print(f'dataset {datasetName} has {datasetsClasses[datasetName]} unique classes')
  return datasetsClasses[datasetName]

def freezeBackbone(backbone):
  for i, param in enumerate(backbone.parameters()):
    param.requires_grad = False

# model = models.resnet50(pretrained=True)
# freezeBackbone(model)

# print(model)


#Move this to outside model...basically want to just get the final weight matrix l2 norm values since it will be score_i which can be used to determine which indices to use to train in second phase
def getScoresAfterTrainingWithGroupLRP(weightsMatrix): #Weights matrix Shape is [out_features, in_features] so norm over out_features since in_features are the large amount of features and out_features is the numClasses
  w_all = weightsMatrix 
  score_i = torch.norm(w_all, p=2, dim=1) #score_i is basically a l2 norm
  return score_i

def getIndicesOfTopFscores(fraction_F, scores):
    num_elements = scores.numel()
    k = int(fraction_F * num_elements)
    _, topFindices = torch.topk(scores, k=k, largest=True)
    return topFindices