#!/bin/bash
#SBATCH --job-name=RunsH2T_EC
#SBATCH --output=job_h2toutput.txt
#SBATCH --error=job_h2terror.txt
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --mem=185Gb
#SBATCH --account=def-eugenium 
#SBATCH --gres=gpu:1 

source /home/humza245/projects/def-eugenium/humza245/torchEnv/bin/activate



# Format for commands python evaluate.py --config_string "learning.learning_rate=0.001, learning.epochs=102, learning.train_batch_size=64, learning.finetune_backbones=False, printTraining=False"


# First job - SVHN
python evaluate.py \
--config_string \
"dataset=SVHN,
 runTypeNameForWandB=H2T_EC,
 learning.useH2T=True,
 learning.use_early_conv_phase=True,
 learning.learning_rate=0.01, 
 learning.fraction_F=0.005, 
 learning.group_lrp_regularizer_coef=0.00001,
 learning.epochs=5, 
 learning.early_conv_epochs=2,
 learning.target_size=512,
 learning.concatLayerSize=106496,
 learning.train_batch_size=128, 
 learning.finetune_backbones=True" &

# # Second job - Flowers102
# python evaluate.py \
# --config_string \
# "dataset=Flowers102,
#  runTypeNameForWandB=H2T_EC,
#  learning.useH2T=True,
#  learning.use_early_conv_phase=True,
#  learning.learning_rate=0.0001, 
#  learning.fraction_F=0.1, 
#  learning.group_lrp_regularizer_coef=0.00001,
#  learning.epochs=50, 
#  learning.early_conv_epochs=5,
#  learning.target_size=512,
#  learning.concatLayerSize=106496,
#  learning.train_batch_size=128, 
#  learning.finetune_backbones=True" &
 
wait
