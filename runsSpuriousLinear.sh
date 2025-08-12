#!/bin/bash
#SBATCH --job-name=RunsSpuriousLinear
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

'''
#First job - Waterbirds 


cp $SCRATCH/waterbird_complete95_forest2water2.tar.gz $SLURM_TMPDIR # This is for waterbirds

tar -xzvf $SLURM_TMPDIR/waterbird_complete95_forest2water2.tar.gz -C $SLURM_TMPDIR # This is for waterbirds

cp $SCRATCH/metadata.csv $SLURM_TMPDIR/waterbird_complete95_forest2water2


python evaluateSpurious.py \
--path $SLURM_TMPDIR --data_path $SLURM_TMPDIR --config_string \
"spuriousConfig.spuriousDataset=Waterbirds,
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
'''
'''
# BEST RESULTS Hyperparameters (wandb name of run: (LIN-BEST)waterbirds (SpuriousLin-With-DFR))
# Test Acc AFTER DFR group 0 = 95.467 group 1 = 95.536 group 2 = 95.81 group 3 = 93.956
# DFRepochs: 5
# epochs: 40
# learning_rate: 0.003
# weight_decay: 0.0004
# batch_size: 32


# MATCHING RESULTS for both before and after DFR Hyperparameters (wandb name of run: (MATCH)waterbirds (SpuriousLin-With-DFR))
# Test Acc AFTER DFR group 0 = 95.47% group 1 = 94.64% group 2 = 94.78% group 3 = 93.96%
# DFRepochs: 5
# epochs: 40
# learning_rate: 0.001
# weight_decay: 0.0004
# batch_size: 32



#Second job - CelebA (much bigger dataset so use larger batch size)

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
 runTypeNameForWandB=SpuriousAffineLin_Seed6,
 spuriousConfig.seed=6,
 learning.useH2T=False,
 learning.useFT_DFR_Phase=False,
 learning.trainOnlyAffineParamOfBNlayers=True,
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
 learning.epochs=6,
 learning.DFRepochs=50,
 spuriousConfig.batch_size=128,
 spuriousConfig.reweight_groups=True, 
 spuriousConfig.pretrained_model=True, 
 spuriousConfig.augment_data=True,  
 spuriousConfig.custom_data_transform=AugWaterbirdsCelebATransform, 
 learning.finetune_backbones=True" &

'''

#Third job - HAM10000

# 2. Copy your dataset on the compute node
# IMPORTANT: Your dataset must be compressed in one single file (zip, hdf5, ...)!!!
cp $SCRATCH/HAM10000.zip $SLURM_TMPDIR # This is for HAM10000 medical dataset

# 3. Eventually unzip your dataset
unzip $SLURM_TMPDIR/HAM10000.zip -d $SLURM_TMPDIR # This is for HAM10000

# Copy metadata to train folder
cp $SCRATCH/ham_metadata.csv $SLURM_TMPDIR/HAM10000/train
#Need to copy to test folder too because of how folders are setup in data file
cp $SCRATCH/ham_metadata.csv $SLURM_TMPDIR/HAM10000/test

python evaluateSpurious.py \
--path $SLURM_TMPDIR --data_path $SLURM_TMPDIR --config_string \
"spuriousConfig.spuriousDataset=HAM10000,
 dataset=HAM10000,
 runTypeNameForWandB=SpuriousLin_Seed1,
 spuriousConfig.seed=1,
 learning.useH2T=False,
 learning.useFT_DFR_Phase=False,
 learning.trainOnlyAffineParamOfBNlayers=False,
 learning.use_early_conv_phase=False,
 learning.optimizer=SGD,
 learning.scheduler=cosine_lr_scheduler,
 learning.learning_rate=0.0003,
 learning.weight_decay=0.0001,
 learning.momentum=0.9,
 learning.DFR_learning_rate=0.0005,
 learning.DFR_weight_decay=0.0004,
 learning.DFR_momentum=0.9,
 learning.DFR_optimizer=SGD,
 learning.epochs=100,
 learning.DFRepochs=500,
 spuriousConfig.batch_size=128,
 spuriousConfig.reweight_groups=True, 
 spuriousConfig.pretrained_model=True, 
 spuriousConfig.augment_data=True,  
 spuriousConfig.custom_data_transform=AugWaterbirdsCelebATransform, 
 learning.finetune_backbones=True" &


'''
#Fourth job - OL3I

# 2. Copy your dataset on the compute node
# IMPORTANT: Your dataset must be compressed in one single file (zip, hdf5, ...)!!!
cp $SCRATCH/OL3I.zip $SLURM_TMPDIR # This is for OL3I medical dataset

# 3. Eventually unzip your dataset
unzip $SLURM_TMPDIR/OL3I.zip -d $SLURM_TMPDIR # This is for OL3I

# Copy metadata file
cp $SCRATCH/ol3i_metadata.csv $SLURM_TMPDIR/OL3I

python evaluateSpurious.py \
--path $SLURM_TMPDIR --data_path $SLURM_TMPDIR --config_string \
"spuriousConfig.spuriousDataset=OL3I,
 dataset=OL3I,
 runTypeNameForWandB=SpuriousLin_Seed1,
 spuriousConfig.seed=1,
 learning.useH2T=False,
 learning.useFT_DFR_Phase=False,
 learning.trainOnlyAffineParamOfBNlayers=False,
 learning.use_early_conv_phase=False,
 learning.optimizer=SGD,
 learning.scheduler=cosine_lr_scheduler,
 learning.learning_rate=0.0001,
 learning.weight_decay=0.0001,
 learning.momentum=0.9,
 learning.DFR_learning_rate=0.0003,
 learning.DFR_weight_decay=0.0001,
 learning.DFR_momentum=0.7,
 learning.DFR_optimizer=SGD,
 learning.epochs=10,
 learning.DFRepochs=200,
 spuriousConfig.batch_size=128,
 spuriousConfig.reweight_groups=True, 
 spuriousConfig.pretrained_model=True, 
 spuriousConfig.augment_data=True,  
 spuriousConfig.custom_data_transform=AugWaterbirdsCelebATransform, 
 learning.finetune_backbones=True" &

'''

wait


