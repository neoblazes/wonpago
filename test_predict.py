# Print out DNN classifier predict for Kifu.
# It works for test.csv and produces human readable output.
#
# Usage: python test_predict.py <model_dir> [kifu_feature_csv]
# Sample usage: python test_predict.py tr0310 small.csv

import numpy as np
import tensorflow as tf

import importlib
import sys

import play_go

if len(sys.argv) < 2:
  print('Usage: python test_predict.py <model_dir> [kifu_feature_csv]')
  exit(1)
model_dir = sys.argv[1]
test_set = sys.argv[2]
print('Test %s file with model in %s' % (test_set, model_dir))

# Load and define features
test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=test_set, target_dtype=np.float32, features_dtype=np.float32, target_column=-1)
x_test, y_test = test_set.data, test_set.target

# Load model and predict
model_fn = importlib.import_module('%s.model_fn' % model_dir)
estimator = model_fn.GetEstimator(model_dir)
ds_predict_tf  = estimator.predict(x_test) 

def GetResultMsg(result):
  if result > 0:
    return 'BLACK'
  elif result < 0:
    return 'WHITE'
  return 'JIGO'

idx = 0
for pred in ds_predict_tf:
  feature = x_test[idx]
  print('%s, predict: %f, real: %s (%f)\n' % (
    play_go.SPrintBoard(feature),
    pred, GetResultMsg(y_test[idx]), y_test[idx]))
  idx = idx + 1
