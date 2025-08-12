import re
from ml_collections import ConfigDict
import os
def get_config(config_string):
    print(f"\n\n\n CONFIG STRING PASSED IS {config_string} \n\n\n")
    train_batch_size = 128
    eval_batch_size = 50
    config = ConfigDict({
        'dataset': 'None',
        'printTraining': True,
        'runTypeNameForWandB': '(No name given to run)',
        'learning': ConfigDict({
            'useH2T': False,
            'useFT_DFR_Phase': False, #If this is True, you can produce spuriousFT(with useH2T False) and spuriousH2T+FT(with useH2T also True). Otherwise SpuriousLin and SpuriousH2T if this is False
            'trainOnlyAffineParamOfBNlayers': False,
            'use_early_conv_phase': False,
            'optimizer': 'SGD',
            'scheduler': "",
            'selectRANDOMfeatures': False,
            'excludeSpuriousFeatureIndices': False,
            'isPredictSpuriousFeaturePhase': False,
            'learning_rate': 0.001,
            'weight_decay': 0.0004,
            'momentum': 0.9,
            'DFR_learning_rate': 0.001,
            'DFR_weight_decay': 0.0004,
            'DFR_momentum': 0.9,
            'DFR_optimizer': 'SGD',
            'fraction_F': 0.1,
            'spuriousFeatFraction_F': 0.01,
            'group_lrp_regularizer_coef': 0.00001,
            'group_lrp_regularizer_r': 2., #NOT using this...these are just to specify to use L2 norm but directly did it
            'group_lrp_regularizer_p': 1., #NOT using this...these are just to specify to use L1 norm but directly did it
            'epochs': 20,
            'DFRepochs': 2,
            'h2tScoreCalcPhaseEpochs': 2,
            'setEarlyLayersScoreToZero': False,
            'num_steps':999999999999, #This is at a max value since only if custom value entered, training will stop updating gradients once number of gradient updates exceeded num_steps
            'early_conv_epochs' : 2, # Early Convergence run to get roughly initialized Linear Layer
            'train_batch_size': train_batch_size,
            'eval_batch_size': eval_batch_size,
            'finetune_backbones': True,
            'target_size': 512,
            'concatLayerSize': 106496 # 4186368 for SVHN, 106496 for Flowers102
        }),
        'model_name': 'Finetune',
        'spuriousConfig': ConfigDict({
            'spuriousDataset': 'Waterbirds',
            'checkpointDirectory': '', #'/home/humza245/projects/def-eugenium/humza245/LabelShiftExperiments/WaterbirdsData/erm_seed1',
            "data_dir_wb": os.path.join(os.environ.get('SLURM_TMPDIR'), 'waterbird_complete95_forest2water2/'),
            "test_wb_dir": os.path.join(os.environ.get('SLURM_TMPDIR'), 'waterbird_complete95_forest2water2/'),
            "data_dir_celebA": os.path.join(os.environ.get('SLURM_TMPDIR'), 'img_align_celeba/'), #Need to add /img_align_celeba/ since The path after unzipping is like /localscratch/humza245.16994749.0/img_align_celeba/img_align_celeba/196975.jpg 
            "test_celebA_dir": os.path.join(os.environ.get('SLURM_TMPDIR'), 'img_align_celeba/'), #Need to add /img_align_celeba/ since The path after unzipping is like /localscratch/humza245.16994749.0/img_align_celeba/img_align_celeba/196975.jpg 
            "data_dir_ham": os.path.join(os.environ.get('SLURM_TMPDIR'), 'HAM10000/'), 
            "test_ham_dir": os.path.join(os.environ.get('SLURM_TMPDIR'), 'HAM10000/'), 
            "data_dir_ol3i": os.path.join(os.environ.get('SLURM_TMPDIR'), 'OL3I/'), 
            "test_ol3i_dir": os.path.join(os.environ.get('SLURM_TMPDIR'), 'OL3I/'),  
            "data_dir_civil": os.path.join(os.environ.get('SLURM_TMPDIR'), 'CivilComments/'), 
            "test_civil_dir": os.path.join(os.environ.get('SLURM_TMPDIR'), 'CivilComments/'), 
            "test_grey_dir": None,
            "test_places_dir": None,
            "output_dir": "/home/humza245/projects/def-eugenium/humza245/LabelShiftExperiments/logs/",
            "pretrained_model": True,
            "reweight_classes": False,
            "reweight_places": False,
            "reweight_groups": False,
            "augment_data": False,
            "custom_data_transform": "default",
            "batch_size": 128,
            'optimizer': 'SGD',
            "eval_freq": 1,
            "seed": 1,
            "multitask": False,
            "predict_place": False,
            "num_minority_groups_remove": 0,
            "resume": None
        })

    })

    '''
    Supported regEx for config_string: 
    custom_config = 'learning_rate=0.01, epochs=100, train_batch_size=64'
    config = get_config(custom_config)
    '''

    if config_string:
        # Example pattern for parsing key-value pairs: key=value
        pattern = r'(\w+(\.\w+)*)\s*=\s*([^,\n]+)'
        matches = re.findall(pattern, config_string)
        for full_key, _, value in matches:
            # Extract the nested key and update the configuration dictionary with parsed values
            keys = full_key.split('.')
            current_dict = config
            for key in keys[:-1]:
                current_dict = current_dict[key]
            last_key = keys[-1]

            # Convert the value to the appropriate data type
            if isinstance(current_dict[last_key], bool):
                if value.lower() == 'true':
                    current_dict[last_key] = True
                else:
                    current_dict[last_key] = False

            elif isinstance(current_dict[last_key], int):
                current_dict[last_key] = int(value)
            elif isinstance(current_dict[last_key], float):
                current_dict[last_key] = float(value)
            elif isinstance(current_dict[last_key], str):
                current_dict[last_key] = str(value)
            # Add more data type conversions for other types as needed

            print(f'Changed config from default for {full_key} to be {current_dict[last_key]}')

    return config
