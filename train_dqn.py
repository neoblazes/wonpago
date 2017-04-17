# Training policy network from Kifu.
# It saaumes that test.csv file is alreay esixting.
# NOTE that it imports model_fn.py from model directory.
#
# Usage: python train_policy.py <dir> <training_csv> <steps>
# Sample usage: python train_policy.py policy_0319 large.csv 100000

import numpy as np
import tensorflow as tf

import argparse
import copy
import csv
import glob
import importlib
import logging
import random
import sys

from tensorflow.python.platform import gfile

import play_go
import train_lib

parser = argparse.ArgumentParser()
parser.add_argument("model_dir")
parser.add_argument("training_csv")
parser.add_argument("steps", type=int)
parser.add_argument("--batch", default=256, type=int)
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--dominion", default=True, type=bool)
parser.add_argument("--learn_value", default=False, type=bool)
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
LEARN_VALUE = args.learn_value

def load_csv(filename):
  line_cnt = sum(1 for row in open(filename))
  with gfile.Open(filename) as csv_file:
    data_file = csv.reader(csv_file)
    data, action, result = [''] * line_cnt, [0] * line_cnt, [0.0] * line_cnt
    idx = 0
    for row in data_file:
      if len(row) >= 81 * 13:
        x, a, r = row[:81 * 13], row[81 * 13], row[81 * 13 + 1]
      else:
        # TODO: deprecated.
        x, y = train_lib.parse_row(row, produce_dominion)
      data[idx] = np.asarray(x, dtype=np.float32)
      action[idx] = int(a)
      result[idx] = float(r)
      idx += 1
    return data, action, result

def load_dataset(pattern):
  files = glob.glob(pattern)
  random.shuffle(files)
  print('Picked csv: %s' % files[0])
  data, action, result = load_csv(files[0])
  if len(files) > 1:
    print('Also pick csv: %s' % files[1])
    data2, action2, result2 = load_csv(files[1])
    return data + data2, action + action2, result + result2
  return data, action, result

# Load network model.
feature, action, result, learn_target = [], [], [], []
print('Working on directory: ', model_dir)
logging.getLogger().setLevel(logging.INFO)
model_fn = importlib.import_module('%s.model_fn' % model_dir)
config = tf.contrib.learn.RunConfig()
config.tf_config.gpu_options.allow_growth=True
estimator = model_fn.GetEstimator(model_dir, config, {"learning_rate": LEARN_RATE})

# Step 1, Read data set. When there are multiple files, select 2 randomly.
print('Loading training data')
feature, action, result = load_dataset(training_csv)

# Step 1, batch retrieve all V'2 (for performance) and build learn target
def RetrieveValues(feature):
  values = [0.0] * len(feature)
  idx = 0
  x_iter = train_lib.xiter(1000)
  pad_size = 1000 - len(feature) % 1000
  x_iter.update(feature + [feature[-1]] * pad_size)
  try:
    predicts = estimator.predict(x=x_iter, outputs=['value'])
    for _ in range(len(feature)):
      values[idx] = predicts.next()['value']
      idx += 1
    return values
  except:
    print('Model is not found, please learn value first.')
    exit(1)  # Comment this line for first run.
    # return (np.random.rand(len(feature)) + 1).tolist()

def BuildQa(result, values):
  # raw_target is [a, R]
  qa = [[1.5]] * (len(result) - 1)
  f = open('Q_a.txt', 'w')
  for i in range(len(result) - 1):  # drop last
    # Set target to V2 when result is 0.
    # If result is not 0, it's terminal and should learn the state itself.
    if result[i] != 0.0:
      qa[i] = [result[i]]
    elif result[i + 1] != 0.0:
      qa[i] = [result[i + 1]]
    else:
      qa[i] = [values[i + 1]]
    f.write('v:%f Q:%f\n' % (float(values[i]), float(qa[i][0])))
  return qa

def MaskAction(action):
  mask = []
  for a in action:
    m = [0] * 82
    if a > 0:
      a = train_lib.PackAction(a)
    m[a] = 1
    mask.append(m)
  return mask[:-1]  # drop last

# Step 2, create training set, Q(a) = V + A(a) = R or V'2
#         training target is [action_mask, target Q(a)]
# Step 3, fit() for training epoches with target V2 (steps = steps/10)
# Step 4, repeat 1~3                                (repeat 10)
def Fit():
  if LEARN_VALUE:
    learn_target = np.concatenate(
        [np.asarray(action, dtype=np.float32).reshape(-1, 1),
         np.asarray(result, dtype=np.float32).reshape(-1, 1)],
        axis=1)
    estimator.fit(x=np.array(feature), y=learn_target, steps=steps, batch_size=BATCH_SIZE)
    return

  mask = MaskAction(action)
  for _ in range(5):
    qa = BuildQa(result, RetrieveValues(feature))
    learn_target = np.concatenate(
        [np.asarray(mask, dtype=np.float32),
         np.asarray(qa, dtype=np.float32)],
        axis=1)
    estimator.fit(x=np.array(feature[:-1]), y=learn_target, steps=steps/5, batch_size=BATCH_SIZE)

Fit()

# Step 5, repeat 1~4 for flips.
# Note that the best Q should be argmax for W, argmin for B.

# Expend to 4 flips.
train_lib.flip_vertical(feature, action)
Fit()

train_lib.flip_horizontal(feature, action)
Fit()

train_lib.flip_vertical(feature, action)
Fit()

# Rotate 90.
train_lib.rot90(feature, action)
Fit()

# Expend to 4 flips with rotated.
train_lib.flip_vertical(feature, action)
Fit()

train_lib.flip_horizontal(feature, action)
Fit()

train_lib.flip_vertical(feature, action)
Fit()
