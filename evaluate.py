
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import time

from absl import app
from absl import flags
from absl import logging

import input_pipeline
#from head2toe import input_pipeline

'''
------------------------------------------------BELOW ARE WHAT THE IMPORT USED TO BE ... MAYBE NEED TO REORGANIZE BUT
FOR TESTING ATM SETTING UP A MODEL FILE WITHOUT MAKING HEAD2TOE AND MODELS FOLDERS


import head2toe.models.finetune as finetune_models
import head2toe.models.finetune_fs as finetune_fs_models

'''

import models.finetune as finetune_models
import models.finetune_fs as finetune_fs_models
from ml_collections import config_flags
import numpy as np


flags.DEFINE_string('output_dir', '/tmp/h2t_evaluation/',
                    'Output directory to dump results.')
FLAGS = flags.FLAGS

config_flags.DEFINE_config_file('config', lock_config=True)


def main(unused_argv):
  config = FLAGS.config
  dataset_name = config.dataset

  image_size = max(config.backbone.input_sizes)
  # The entire dataset is the episode for VTAB.
  # We calculate average accuracy if is validation.
  # If the crossvalidation is not enabled, then we perform only 5th fold,
  # which is the default validation set.
  if config.is_vtab_5fold_valid and config.eval_mode == 'valid':
    start_idx = 0
  else:
    start_idx = 4
  end_idx = 5

  if config.model_name == 'Finetune':
    model = finetune_models.Finetune(config)
  elif config.model_name == 'FinetuneFS':
    model = finetune_fs_models.FinetuneFS(config)
  else:
    raise ValueError(f'config.model_name: {config.model_name} not valid.')

  episode_metrics = []
  start_time = time.time()
  for episode_idx in range(start_idx, end_idx):
    t0 = time.perf_counter()
    input_fn = functools.partial(
        input_pipeline.create_vtab_dataset, config.dataset,
        image_size=image_size, eval_mode=config.eval_mode,
        valid_fold_id=episode_idx)
    if config.learning.data_fraction != 1:
      logging.info('Fractional data: %f', config.learning.data_fraction)
      support_dataset = input_pipeline.create_vtab_dataset_balanced(
          config.dataset, image_size=image_size,
          batch_size=config.learning.train_batch_size,
          data_fraction=config.learning.data_fraction)
    else:
      support_dataset = input_fn(mode='train',
                                 batch_size=config.learning.train_batch_size)
    query_dataset = input_fn(mode='eval',
                             batch_size=config.learning.eval_batch_size)
    if (hasattr(config.learning, 'feature_selection') and
        config.learning.feature_selection.fs_dataset):
      input_fn = functools.partial(
          input_pipeline.create_vtab_dataset,
          config.learning.feature_selection.fs_dataset,
          image_size=image_size, eval_mode=config.eval_mode,
          valid_fold_id=episode_idx)
      fs_dataset = input_fn(
          mode='train', batch_size=config.learning.train_batch_size)
    else:
      fs_dataset = None
    metrics = model.evaluate(config.learning,
                             support_dataset,
                             query_dataset,
                             fs_dataset=fs_dataset)
    episode_metrics.append(metrics)
    logging.info('Episode time %f:', time.perf_counter() - t0)
    logging.info('Episode: %i', episode_idx)
    logging.info('Final support loss: %f', metrics['support_loss'])
    logging.info('Final support accuracy: %4.2f%%',
                 100 * metrics['support_accuracy'])
    logging.info('Final query loss: %f', metrics['query_loss'])
    logging.info('Final query accuracy: %4.2f%%',
                 100 * metrics['query_accuracy'])

    logging.info('Total iteration time %f:', time.perf_counter() - t0)


  logging.info('Elapsed time %f:', time.time() - start_time)
  all_q_accs = [m['query_accuracy'] for m in episode_metrics]
  acc_mean, acc_std = np.mean(all_q_accs), np.std(all_q_accs)
  logging.info('Query Accucary Mean: %.2f', acc_mean)
  logging.info('Query Accucary Std: %.2f', acc_std)

if __name__ == '__main__':
  app.run(main)