# LabelShiftExperiments

evaluateSpurious.py serves as the main starting point of execution. Shell files explain setup and other details for executing the code with computation resources


ABSTRACT
Spurious correlations are a major source of errors for machine learning models, in particular
when aiming for group-level fairness. It has been recently shown that a powerful approach to
combat spurious correlations is to re-train the last layer on a balanced validation dataset, isolating
robust features for the predictor. However, key attributes can sometimes be discarded by neural
networks towards the last layer. In this work, we thus consider retraining a classifier on a set of
features derived from all layers. We utilize a recently proposed feature selection strategy to select
unbiased features from all the layers. We observe this approach gives significant improvements
in worst-group accuracy on several standard benchmarks. Another pain point in transfer learning
is with out-of-distribution tasks having large distribution shifts relative to the source task. Full
finetuning suffers in performance as it disturbs backbone parameter weights during the starting
few optimization steps and is forced to make drastic adaptations to correct for large losses initially
observed in training. Linear tuning is another approach shown to improve model generalization
capabilities and is especially effective for transfer learning on out-of-distribution downstream tasks.
We further evaluate the usefulness of intermediate layer information by incorporating it with a linear
tuning approach. Results over datasets from a common visual task adaptation benchmark show that
the empirical benefits from simply leveraging intermediate layers are similar to the proposed method
and there is no noticeable gain in accuracy from incorporating a linear tuning step.

-----------------------------------------------------------------------------------------------------------------
The code provided is in support of my thesis for the Master of Computer Science at Concordia University. 

Full thesis report is available on request to humzaw28@gmail.com

Available to public - 
ECCV 2024 Workshop submission: https://arxiv.org/pdf/2409.14637
The contributions of Chapter 3 of the thesis have been presented at the ECCV 2024 Fairness and ethics
towards transparent AI: facing the chalLEnge through model Debiasing (FAILED) workshop on
July 31, 2024: Not Only the Last-Layer Features for Spurious Correlations: All Layer Deep Feature
Reweighting by Humza Wajid Hameed, Geraldin Nanfack, Eugene Belilovsky

-----------------------------------------------------------------------------------------------------------------
