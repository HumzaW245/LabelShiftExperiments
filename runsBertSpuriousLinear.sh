#!/bin/bash
#SBATCH --job-name=RunsBertSpuriousLinear
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --ntasks=1
#SBATCH --time=15:00:00
#SBATCH --mem=64Gb
#SBATCH --account=def-eugenium 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

source /home/humza245/projects/def-eugenium/humza245/deep_feature_reweighting/torchDFRenv/bin/activate
export WANDB_MODE=online

#Go into wandb folder and run 'wandb sync /' to sync offline logs to online repo


# note: Need to move dataset zip file to scratch first by doing e.g. scp waterbird_complete95_forest2water2.tar.gz /home/humza245/scratch
# note: After the above, need to also move the metadata file e.g. scp metadata.csv /home/humza245/scratch
# Then the below steps will copy from scratch to temp directory in slurm for faster running, unzip it
# and then after copying the metadata csv within the unzipped folder (e.g. cp $SCRATCH/celeba_metadata.csv $SLURM_TMPDIR/img_align_celeba),  
# program can then run (See examples of setups below)

# Format for commands python evaluate.py --config_string "learning.learning_rate=0.001, learning.epochs=102, learning.train_batch_size=64, learning.finetune_backbones=False, printTraining=False"


#First job - CivilComments 


cp $SCRATCH/CivilComments $SLURM_TMPDIR 

python evaluateBertSpurious.py \
--path $SLURM_TMPDIR --data_path $SLURM_TMPDIR --config_string \
"spuriousConfig.spuriousDataset=CivilComments,
 dataset=Waterbirds,
 runTypeNameForWandB=SpuriousLin_Seed6,
 spuriousConfig.seed=6,
 learning.useH2T=False,
 learning.useFT_DFR_Phase=False,
 learning.trainOnlyAffineParamOfBNlayers=False,
 learning.use_early_conv_phase=False,
 learning.optimizer=SGD,
 learning.scheduler=cosine_lr_scheduler,
 learning.learning_rate=0.003,
 learning.weight_decay=0.0004,
 learning.momentum=0.9,
 learning.DFR_learning_rate=0.0001,
 learning.DFR_weight_decay=0.0001,
 learning.DFR_momentum=0.9,
 learning.DFR_optimizer=SGD,
 learning.epochs=20,
 learning.DFRepochs=100,
 spuriousConfig.batch_size=32,
 spuriousConfig.reweight_groups=True, 
 spuriousConfig.pretrained_model=True, 
 spuriousConfig.augment_data=True,  
 spuriousConfig.custom_data_transform=AugWaterbirdsCelebATransform, 
 learning.finetune_backbones=True" &


wait


