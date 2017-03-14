# Training Estimator from Kifu.
# It saaumes that test.csv file is alreay esixting.
# NOTE that it imports model_fn.py from model directory.
#
# Usage: python train_kifu.py <dir> [steps]
# Sample usage: python train_kifu.py dnnregressor test.csv 2000

import numpy as np
import tensorflow as tf

import importlib
import logging
import sys

if len(sys.argv) < 4:
  print('Usage: python train_kifu.py <dir> <training_csv> <steps>')
  exit(1)

model_dir=sys.argv[1]
training_csv=sys.argv[2]
steps=int(sys.argv[3])

# Load network model.
print('Working on directory: ', model_dir)
model_fn = importlib.import_module('%s.model_fn' % model_dir)
config = tf.contrib.learn.RunConfig()
config.tf_config.gpu_options.allow_growth=True
estimator = model_fn.GetEstimator(model_dir, config)

# Read data set
def flip_vertical(feature):
  for i in range(4):
    feature[i*9:i*9+9], feature[81-i*9-9:81-i*9] = feature[81-i*9-9:81-i*9], feature[i*9:i*9+9]
    # Also flip liberty
    if len(feature) > 85:
      feature[i*9+85:i*9+9+85], feature[85+81-i*9-9:85+81-i*9] = feature[85+81-i*9-9:85+81-i*9], feature[i*9+85:i*9+9+85]

def flip_horizontal(feature):
  for i in range(9):
    feature[i*9:i*9+9] = np.flipud(feature[i*9:i*9+9])
    # Also flip liberty
    if len(feature) > 85:
      feature[i*9+85:i*9+9+85] = np.flipud(feature[i*9+85:i*9+9+85])

training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=training_csv, target_dtype=np.float32, features_dtype=np.float32, target_column=-1)
x_train, y_train = training_set.data, training_set.target
# Expend to 4 flips.
x_train_fv = x_train.copy()
flip_vertical(x_train_fv)
x_train_fh = x_train.copy()
flip_horizontal(x_train_fh)
x_train_fa = x_train_fh.copy()
flip_vertical(x_train_fa)
x_train = np.concatenate((x_train, x_train_fv, x_train_fh, x_train_fa), axis=0)
y_train = np.concatenate(([y_train] * 4), axis=0)

# Training
logging.getLogger().setLevel(logging.INFO)
estimator.fit(x=x_train, y=y_train, steps=steps)
