# Training policy network from Kifu.
# It saaumes that test.csv file is alreay esixting.
# NOTE that it imports model_fn.py from model directory.
#
# Usage: python train_policy.py <dir> <training_csv> <steps>
# Sample usage: python train_policy.py policy_0319 large.csv 100000

import numpy as np
import tensorflow as tf

import argparse
import csv
import importlib
import logging
import sys

from tensorflow.python.platform import gfile

import play_go
import train_lib

parser = argparse.ArgumentParser()
parser.add_argument("model_dir")
parser.add_argument("training_csv")
parser.add_argument("steps", type=int)
parser.add_argument("--batch", default=256, type=int)
parser.add_argument("--lr", default=0.00001, type=float)
parser.add_argument("--dominion", default=True, type=bool)
args = parser.parse_args()
model_dir=args.model_dir
training_csv=args.training_csv
steps=args.steps
BATCH_SIZE=args.batch
LEARN_RATE=args.lr
produce_dominion=args.dominion
num_feature = 9
if produce_dominion:
  num_feature += 4

def load_dataset(filename):
  with gfile.Open(filename) as csv_file:
    data_file = csv.reader(csv_file)
    data, target = [], []
    for row in data_file:
      if len(row) >= 81 * 13:
        x, y = row[:81 * 13], int(row[-1])
      else:
        x, y = train_lib.parse_row(row, produce_dominion)
      data.append(np.asarray(x, dtype=np.float32))
      target.append(y)
    return data, target

# Load network model.
feature, target = ([], [])
print('Working on directory: ', model_dir)
logging.getLogger().setLevel(logging.INFO)
model_fn = importlib.import_module('%s.model_fn' % model_dir)
config = tf.contrib.learn.RunConfig()
config.tf_config.gpu_options.allow_growth=True
estimator = model_fn.GetEstimator(model_dir, config, {"learning_rate": LEARN_RATE})
def Fit():
  estimator.fit(x=np.array(feature), y=train_lib.target_nparray(target), steps=steps, batch_size=BATCH_SIZE)
  pass

# Read data set.
print('Loading training data')
feature, target = load_dataset(training_csv)
Fit()

# Expend to 4 flips.
train_lib.flip_vertical(feature, target)
Fit()

train_lib.flip_horizontal(feature, target)
Fit()

train_lib.flip_vertical(feature, target)
Fit()

# Rotate 90.
train_lib.rot90(feature, target)
Fit()

# Expend to 4 flips with rotated.
train_lib.flip_vertical(feature, target)
Fit()

train_lib.flip_horizontal(feature, target)
Fit()

train_lib.flip_vertical(feature, target)
Fit()
