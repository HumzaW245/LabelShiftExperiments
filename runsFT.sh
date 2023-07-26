#!/bin/bash
#SBATCH --job-name=RunsFT
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
# python evaluate.py \
# --config_string \
# "dataset=SVHN,
#  runTypeNameForWandB=FT,
#  learning.useH2T=False,
#  learning.use_early_conv_phase=False
#  learning.learning_rate=0.01, 
#  learning.epochs=5, 
#  learning.train_batch_size=128, 
#  learning.finetune_backbones=True" &

# Second job - Flowers102
python evaluate.py \
--config_string \
"dataset=Flowers102,
 runTypeNameForWandB=FT,
 learning.useH2T=False,
 learning.use_early_conv_phase=False
 learning.learning_rate=0.0001, 
 learning.epochs=50, 
 learning.train_batch_size=128, 
 learning.finetune_backbones=True" &
 
wait
