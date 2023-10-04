#!/bin/bash
#SBATCH --job-name=RunsSpuriousLinear
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --mem=64Gb
#SBATCH --account=def-eugenium 
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=4

source /home/humza245/projects/def-eugenium/humza245/deep_feature_reweighting/torchDFRenv/bin/activate


# Format for commands python evaluate.py --config_string "learning.learning_rate=0.001, learning.epochs=102, learning.train_batch_size=64, learning.finetune_backbones=False, printTraining=False"


#First job - Waterbirds 
python evaluateSpurious.py \
--config_string \
"dataset=waterbirds
 runTypeNameForWandB=SpuriousLin-With-DFRdoneOnValLoader,
 learning.useH2T=False,
 learning.use_early_conv_phase=False,
 learning.optimizer=SGD,
 learning.scheduler=cosine_lr_scheduler,
 learning.learning_rate=0.003,
 learning.weight_decay=0.0004,
 learning.epochs=40,
 learning.DFRepochs=5,
 spuriousConfig.batch_size=32, 
 spuriousConfig.reweight_groups=True, 
 spuriousConfig.pretrained_model=True, 
 spuriousConfig.augment_data=True,  
 spuriousConfig.custom_data_transform=AugWaterbirdsCelebATransform, 
 learning.finetune_backbones=True" &

# BEST RESULTS Hyperparameters (wandb name of run: (BEST)waterbirds (SpuriousLin-With-DFR))
# Test Acc AFTER DFR group 0 = 95.467 group 1 = 95.536 group 2 = 95.81 group 3 = 93.956
# DFRepochs: 5
# epochs: 40
# learning_rate: 0.003
# weight_decay: 0.0004
# batch_size: 32


# MATCHING RESULTS for both before and after DFR Hyperparameters (wandb name of run: (MATCH)waterbirds (SpuriousLin-With-DFR))
#Test Acc AFTER DFR group 0 = 95.47% group 1 = 94.64% group 2 = 94.78% group 3 = 93.96%
# DFRepochs: 5
# epochs: 40
# learning_rate: 0.001
# weight_decay: 0.0004
# batch_size: 32


wait


