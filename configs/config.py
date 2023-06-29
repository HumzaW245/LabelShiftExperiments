import re
from ml_collections import ConfigDict

def get_config(config_string):
    train_batch_size = 128
    eval_batch_size = 50
    config = ConfigDict({
        'dataset': 'SVHN',
        'printTraining': True,
        'runTypeNameForWandB': '(No name given to run)',
        'learning': ConfigDict({
            'useH2T': True,
            'optimizer': 'adam',
            'learning_rate': 0.1,
            'weight_decay': 0,
            'momentum': 0,
            #'l1_regularizer': 0.,
            #'l2_regularizer': 0.,
            'epochs': 3,
            'train_batch_size': train_batch_size,
            'eval_batch_size': eval_batch_size,
            'finetune_backbones': False,
            'target_size': 24576,
            'concatLayerSize': 4186368,
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

    return config
