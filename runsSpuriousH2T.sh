#!/bin/bash
#SBATCH --job-name=RunsSpuriousH2T
#SBATCH --output=job_output_h2t.txt
#SBATCH --error=job_error_h2t.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=128Gb
#SBATCH --account=def-eugenium 
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=4

source /home/humza245/projects/def-eugenium/humza245/deep_feature_reweighting/torchDFRenv/bin/activate


# Format for commands python evaluate.py --config_string "learning.learning_rate=0.001, learning.epochs=102, learning.train_batch_size=64, learning.finetune_backbones=False, printTraining=False"

# #First job - Waterbirds 
# python evaluateSpurious.py \
# --config_string \
# "dataset=waterbirds,
#  ru==========FillUsingCelebA================as copy initial start point=True, 
#  spuriousConfig.pretrained_model=True, 
#  spuriousConfig.augment_data=True, 
#  spuriousConfig.custom_data_transform=AugWaterbirdsCelebATransform, 
#  learning.finetune_backbones=True" &



#Second job - CelebA 


# 2. Copy your dataset on the compute node
# IMPORTANT: Your dataset must be compressed in one single file (zip, hdf5, ...)!!!
cp $SCRATCH/archive.zip $SLURM_TMPDIR # This is for celebA

# 3. Eventually unzip your dataset
unzip $SLURM_TMPDIR/archive.zip -d $SLURM_TMPDIR # This is for celebA

# Copy metadata to img_align_celeba with tempdir since unzipping creates 2 img_align_celba subfolders (See config.py 'data_dir_celebA')
cp $SCRATCH/celeba_metadata.csv $SLURM_TMPDIR/img_align_celeba

python evaluateSpurious.py \
--path $SLURM_TMPDIR --data_path $SLURM_TMPDIR --config_string \
"spuriousConfig.spuriousDataset=CelebA,
 dataset=CelebA,
 runTypeNameForWandB=SpuriousH2T_Seed5,
 spuriousConfig.seed=5,
 learning.useH2T=True,
 learning.use_early_conv_phase=False,
 learning.optimizer=SGD,
 learning.scheduler=cosine_lr_scheduler,
 learning.learning_rate=0.0005,
 learning.weight_decay=0.0001,
 learning.momentum=0.9,
 learning.DFR_learning_rate=0.0001,
 learning.DFR_weight_decay=0.0001,
 learning.DFR_momentum=0.4,
 learning.DFR_optimizer=SGD,
 learning.fraction_F=0.56, 
 learning.group_lrp_regularizer_coef=0.0009,
 learning.epochs=6,
 learning.DFRepochs=50,
 learning.h2tScoreCalcPhaseEpochs=15,
 learning.setEarlyLayersScoreToZero=False,
 learning.early_conv_epochs=2,
 learning.target_size=512,
 learning.concatLayerSize=102406,
 spuriousConfig.batch_size=128, 
 spuriousConfig.reweight_groups=True, 
 spuriousConfig.pretrained_model=True, 
 spuriousConfig.augment_data=True, 
 spuriousConfig.custom_data_transform=AugWaterbirdsCelebATransform, 
 learning.finetune_backbones=True" &

wait


#USING BIG Fraction like = 0.5 much better results (so what Eugene said was correct. Try other stuff like this)
