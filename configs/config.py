

import re
from ml_collections import ConfigDict

def get_config(config_string):
  train_batch_size = 128
  eval_batch_size = 50
  config = ConfigDict({
      'dataset':
          'SVHN',
      'runTypeNameForWandB':
          '(No name given to run)',
                #Currently not being used
                '''Currently not being used
                'eval_mode':
                    'valid',
                'is_vtab_5fold_valid':
                    True,
                'seed':
                    8,
                'max_num_gpus':
                    1,
                '''
      'learning': #COMMENTED OUT UNUSED ONES 
          ConfigDict({
              'optimizer': 'adam',  #  adadelta, adadelta_adaptive, sgd
              'learning_rate': 0.1,
              #'grad_clip_value': -1.,  # Applied if positive.
              'l1_regularizer': 0.,
              'l2_regularizer': 0.,
              'group_lrp_regularizer_coef': 0.,
              'group_lrp_regularizer_r': 2.,
              'group_lrp_regularizer_p': 1.,
              'group_lrp_is_embedding': False,
              'epochs': 500,
              'data_fraction': 1.,
              #'cached_eval': True,
              #'use_cosine_decay': True,
              'train_batch_size': train_batch_size,
              'eval_batch_size': eval_batch_size,
              'finetune_backbones': False,
              'finetune_lr_multiplier': 1.,
              'finetune_steps_multiplier': 1.,
              # ('', 'unit_vector', 'per_feature')
              'feature_normalization': 'unit_vector',
              # nohidden, random_100, random_1000, trainable_100, trainable_1000
              #'output_head_type': 'nohidden',
              #'output_head_zeroinit': False,
              'log_freq': 50,
              #NEW CODE ADDED FOR MULTIPLE PHASE TRAINING
              'is2PhaseMethod': False,
              'useEarlyConvergePhase1': False,
              'phase1TrainingSteps': 500
          }),
      'model_name':
          'Finetune'
  })

  return config
