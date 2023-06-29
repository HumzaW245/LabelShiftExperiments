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
def getScoresAfterTrainingWithGroupLRP(device, weightsMatrix): #Weights matrix Shape is [out_features, in_features] so norm over out_features (dim = 0) since in_features are the large amount of features and out_features is the numClasses
  weightsMatrix = weightsMatrix.to(device)
  w_all = weightsMatrix 
  score_i = torch.norm(w_all, p=2, dim=0) #score_i is basically a l2 norm
  return score_i

def getIndicesOfTopFscores(device, fraction_F, scores):
    scores = scores.to(device)
    #print(f'IN getIndicesOfTopFscores This the scores matrix shape after phase 1: {scores.shape} and type of score is {type(scores)}') #Take indices of top F% and pass as indices in 2nd phase
    num_elements = scores.numel()
    k = int(fraction_F * num_elements)
    print(f'fraction F means k top features to get where k = {k}')
    _, topFindices = torch.topk(scores, k=k, largest=True)
    #print('before returning topFindices')
    print(f'topF percent indices = {len(topFindices)} and the indices are {topFindices}')
    return topFindices

def getOptimizer(model, learningConfig):
  
  if learningConfig.optimizer == 'adam':
    optimizer = torch.optim.Adam(
      model.parameters(), 
      lr=learningConfig.learning_rate, 
      weight_decay=learningConfig.weight_decay
      )
  elif learningConfig.optimizer == 'SGD':
    optimizer = torch.optim.SGD(
      model.parameters(), 
      lr=learningConfig.learning_rate, 
      weight_decay=learningConfig.weight_decay, 
      momentum=learningConfig.momentum
      )

  else:
    raise ValueError("the config optimizer used is not supported. Needs to be defined where others are defined like SGD and adam.")

  return optimizer