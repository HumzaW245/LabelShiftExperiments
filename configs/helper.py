import torch
import matplotlib.pyplot as plt
def numUniqueClasses(datasetName):

  datasetsClasses = {'SVHN': 10, 'Flowers102': 102, 'EuroSAT': 10}
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

def layersForTopFPctIndicesSelected(selected_feature_indices, layersWithRangesDict):
  result = {}
  print("inlayers func")
  for key, ranges in layersWithRangesDict.items():
      count = 0

      start = ranges[0]
      end = ranges[1]
      
      for index in selected_feature_indices:
          if start <= index <= end:
              count += 1
      result[key] = count / len(selected_feature_indices) # % in each key
  
  return result  


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


  import matplotlib.pyplot as plt

def plotLayersSelectedFeaturesPct(layersUsedForTopFPctIndicesSelected):
  data = layersUsedForTopFPctIndicesSelected
  
  keys = list(data.keys())
  values = list(data.values())


  plt.figure(figsize=(32, 16))  # Set the size of the figure
  plt.bar(range(len(keys)), values)  # Create a bar plot
  plt.xticks(range(len(keys)), keys, rotation=90)  # Set the x-axis labels

  plt.xlabel('Layers')  # Set the x-axis label
  plt.ylabel('Values')  # Set the y-axis label
  plt.title(f'The top {learningConfig.fraction_F * 100} % features selected by layer')  # Set the title of the plot
  
  # Save the plot as an image file
  plt.savefig('layersUsedForTopFPctIndicesSelected.png')

