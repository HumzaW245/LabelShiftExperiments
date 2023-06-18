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