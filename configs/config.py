import re
from ml_collections import ConfigDict

def get_config(config_string):
    train_batch_size = 128
    eval_batch_size = 50
    config = ConfigDict({
        'dataset': 'Flowers102',
        'printTraining': True,
        'runTypeNameForWandB': '(No name given to run)',
        'learning': ConfigDict({
            'useH2T': False,
            'use_early_conv_phase': False,
            'optimizer': 'adam',
            'learning_rate': 0.0001,
            'weight_decay': 0,
            'momentum': 0,
            'fraction_F': 0.1,
            'group_lrp_regularizer_coef': 0.00001,
            'group_lrp_regularizer_r': 2., #NOT using this...these are just to specify to use L2 norm but directly did it
            'group_lrp_regularizer_p': 1., #NOT using this...these are just to specify to use L1 norm but directly did it
            'epochs': 50,
            'early_conv_epochs' : 2, # Early Convergence run to get roughly initialized Linear Layer
            'train_batch_size': train_batch_size,
            'eval_batch_size': eval_batch_size,
            'finetune_backbones': True,
            'target_size': 512,
            'concatLayerSize': 106496, # 4186368 for SVHN, 106496 for Flowers102
            #'finetune_lr_multiplier': 1.,
            #'finetune_steps_multiplier': 1.,
            #'feature_normalization': 'unit_vector',
            #'log_freq': 50,
            #'is2PhaseMethod': False,
            #'useEarlyConvergePhase1': False,
            'phase1TrainingSteps': 500
        }),
        'model_name': 'Finetune'
    })

    '''
    Supported regEx for config_string: 
    custom_config = 'learning_rate=0.01, epochs=100, train_batch_size=64'
    config = get_config(custom_config)
    '''

    if config_string:
        # Example pattern for parsing key-value pairs: key=value
        pattern = r'(\w+)\s*=\s*([^,\n]+)'
        matches = re.findall(pattern, config_string)
        for key, value in matches:
            # Update the configuration dictionary with parsed values
            config[key] = value
            print(f'Changed config from default for {key} to be {config[key]}')

    return config
