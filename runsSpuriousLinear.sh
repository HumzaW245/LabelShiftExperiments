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
 learning.learning_rate=0.0005,
 learning.weight_decay=0.0004,
 learning.epochs=40,
 learning.DFRepochs=5,
 spuriousConfig.batch_size=32, 
 spuriousConfig.reweight_groups=True, 
 spuriousConfig.pretrained_model=True, 
 spuriousConfig.augment_data=True,  
 spuriousConfig.custom_data_transform=AugWaterbirdsCelebATransform, 
 learning.finetune_backbones=True" &

wait


