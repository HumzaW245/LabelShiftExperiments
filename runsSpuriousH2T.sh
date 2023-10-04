#!/bin/bash
#SBATCH --job-name=RunsSpuriousH2T
#SBATCH --output=job_output_h2t.txt
#SBATCH --error=job_error_h2t.txt
#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --mem=128Gb
#SBATCH --account=def-eugenium 
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=4

source /home/humza245/projects/def-eugenium/humza245/deep_feature_reweighting/torchDFRenv/bin/activate


# Format for commands python evaluate.py --config_string "learning.learning_rate=0.001, learning.epochs=102, learning.train_batch_size=64, learning.finetune_backbones=False, printTraining=False"

#First job - Waterbirds 
python evaluateSpurious.py \
--config_string \
"dataset=waterbirds,
 runTypeNameForWandB=SpuriousH2T-With-DFR_FTon,
 learning.useH2T=True,
 learning.use_early_conv_phase=False,
 learning.optimizer=SGD,
 learning.scheduler=cosine_lr_scheduler,
 learning.learning_rate=0.0005,
 learning.weight_decay=0.0004,
 learning.fraction_F=0.75, 
 learning.group_lrp_regularizer_coef=0.0005,
 learning.epochs=50,
 learning.DFRepochs=70,
 learning.h2tScoreCalcPhaseEpochs=15,
 learning.early_conv_epochs=2,
 learning.target_size=512,
 learning.concatLayerSize=102406,
 spuriousConfig.batch_size=32, 
 spuriousConfig.reweight_groups=True, 
 spuriousConfig.pretrained_model=True, 
 spuriousConfig.augment_data=True, 
 spuriousConfig.custom_data_transform=AugWaterbirdsCelebATransform, 
 learning.finetune_backbones=True" &

wait


#USING BIG Fraction like = 0.5 much better results (so what Eugene said was correct. Try other stuff like this)
