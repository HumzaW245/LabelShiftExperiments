#!/bin/bash
#SBATCH --job-name=RunsSpuriousH2T
#SBATCH --output=job_output_h2t.txt
#SBATCH --error=job_error_h2t.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --mem=128Gb
#SBATCH --account=def-eugenium 
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=4

source /home/humza245/projects/def-eugenium/humza245/deep_feature_reweighting/torchDFRenv/bin/activate
export WANDB_MODE=online
#Go into wandb folder and run 'wandb sync /' to sync offline logs to online repo

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
 runTypeNameForWandB=Target8192(F_0.05)SpuriousH2T_Seed5(valRWforFeatSelect),
 spuriousConfig.seed=5,
 learning.useH2T=True,
 learning.useFT_DFR_Phase=False,
 learning.trainOnlyAffineParamOfBNlayers=False,
 learning.excludeSpuriousFeatureIndices=False,
 learning.use_early_conv_phase=False,
 learning.optimizer=SGD,
 learning.scheduler=cosine_lr_scheduler,
 learning.selectRANDOMfeatures=False,
 learning.learning_rate=0.0005,
 learning.weight_decay=0.0004,
 learning.momentum=0.9,
 learning.DFR_learning_rate=0.0005,
 learning.DFR_weight_decay=0.0003,
 learning.DFR_momentum=0.9,
 learning.DFR_optimizer=SGD,
 learning.fraction_F=0.05, 
 learning.spuriousFeatFraction_F=0.01, 
 learning.group_lrp_regularizer_coef=0.00001,
 learning.epochs=70,
 learning.DFRepochs=500,
 learning.h2tScoreCalcPhaseEpochs=25,
 learning.setEarlyLayersScoreToZero=False,
 learning.early_conv_epochs=2,
 learning.target_size=8192,
 learning.concatLayerSize=1355526,
 spuriousConfig.batch_size=32, 
 spuriousConfig.reweight_groups=True, 
 spuriousConfig.pretrained_model=True, 
 spuriousConfig.augment_data=True, 
 spuriousConfig.custom_data_transform=AugWaterbirdsCelebATransform, 
 learning.finetune_backbones=True" &
'''
'''
#Second job - CelebA 


# 2. Copy your dataset on the compute node
# IMPORTANT: Your dataset must be compressed in one single file (zip, hdf5, ...)!!!
cp $SCRATCH/archive.zip $SLURM_TMPDIR # This is for celebA

# 3. Eventually unzip your dataset
unzip $SLURM_TMPDIR/archive.zip -d $SLURM_TMPDIR # This is for celebA

# Copy metadata to img_align_celeba with tempdir since unzipping creates 2 img_align_celba subfolders (See config.py 'data_dir_celebA')
cp $SCRATCH/celeba_metadata.csv $SLURM_TMPDIR/img_align_celeba

#NOTE: For now, concatLayerSize may need to be set based on error. e.g. if target size is 8192, error will show that matmul is expeting 1355526 neurons incoming so set it to that
# Target size 512 - use concatLayerSize 102406
# Target size 8192 - use concatLayerSize 1355526
python evaluateSpurious.py \
--path $SLURM_TMPDIR --data_path $SLURM_TMPDIR --config_string \
"spuriousConfig.spuriousDataset=CelebA,
 dataset=CelebA,
 runTypeNameForWandB=(F_0.05)SpuriousH2T_Seed5(valRWforFeatSelect),
 spuriousConfig.seed=5,
 learning.useH2T=True,
 learning.useFT_DFR_Phase=False,
 learning.trainOnlyAffineParamOfBNlayers=False,
 learning.excludeSpuriousFeatureIndices=False,
 learning.use_early_conv_phase=False,
 learning.optimizer=SGD,
 learning.scheduler=cosine_lr_scheduler,
 learning.selectRANDOMfeatures=False,
 learning.learning_rate=0.0005,
 learning.weight_decay=0.0001,
 learning.momentum=0.9,
 learning.DFR_learning_rate=0.0005,
 learning.DFR_weight_decay=0.0003,
 learning.DFR_momentum=0.9,
 learning.DFR_optimizer=SGD,
 learning.fraction_F=0.05, 
 learning.spuriousFeatFraction_F=0.01, 
 learning.group_lrp_regularizer_coef=0.00001,
 learning.epochs=20,
 learning.DFRepochs=300,
 learning.h2tScoreCalcPhaseEpochs=25,
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
 runTypeNameForWandB=(F_0.05)SpuriousH2T_Seed3(valRWforFeatSelect),
 spuriousConfig.seed=3,
 learning.useH2T=True,
 learning.useFT_DFR_Phase=False,
 learning.trainOnlyAffineParamOfBNlayers=False,
 learning.excludeSpuriousFeatureIndices=False,
 learning.use_early_conv_phase=False,
 learning.optimizer=SGD,
 learning.scheduler=cosine_lr_scheduler,
 learning.selectRANDOMfeatures=False,
 learning.learning_rate=0.0003,
 learning.weight_decay=0.0001,
 learning.momentum=0.9,
 learning.DFR_learning_rate=0.0005,
 learning.DFR_weight_decay=0.0004,
 learning.DFR_momentum=0.9,
 learning.DFR_optimizer=SGD,
 learning.fraction_F=0.05, 
 learning.spuriousFeatFraction_F=0.01, 
 learning.group_lrp_regularizer_coef=0.0001,
 learning.epochs=100,
 learning.DFRepochs=500,
 learning.h2tScoreCalcPhaseEpochs=100,
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
 runTypeNameForWandB=SpuriousH2T_Seed1(valRWforFeatSelect),
 spuriousConfig.seed=1,
 learning.useH2T=True,
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
 learning.fraction_F=0.56, 
 learning.group_lrp_regularizer_coef=0.0009,
 learning.epochs=10,
 learning.DFRepochs=200,
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

'''

wait


#USING BIG Fraction like = 0.5 much better results (so what Eugene said was correct. Try other stuff like this)
