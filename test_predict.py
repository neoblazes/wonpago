# Print out DNN classifier predict for Kifu.
# It works for test.csv and produces human readable output.
#
# Usage: python test_predict.py <model_dir> [kifu_feature_csv]
# Sample usage: python test_predict.py tr0310 small.csv

import numpy as np
import tensorflow as tf

import importlib
import sys

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

# Print out human readable.
BOARD_CHAR = { -1: 'O', 1: '@', 0: '.' }
TURN_MSG = { 1: 'BLACK(@)', -1: 'WHITE(O)' }

def GetResultMsg(result):
  if result > 0:
    return 'BLACK'
  elif result < 0:
    return 'WHITE'
  return 'JIGO'
  
idx = 0
for pred in ds_predict_tf:
  feature = x_test[idx]
  board = feature[:81]
  last_move = feature[81]
  ko_pos = feature[82:84]
  pos = 0
  for row in range(1, 10):
    outstr = ''
    for col in range(1, 10):
      if row == ko_pos[0] and col == ko_pos[1]:
        outstr = outstr + '*'
      else:
        outstr = outstr + BOARD_CHAR[board[pos]]
      pos = pos + 1
    print(outstr)
  print('Last move %s, predict: %f, real: %s (%f)\n' %
        (TURN_MSG[last_move], pred - 1, GetResultMsg(y_test[idx]), y_test[idx]))
  idx = idx + 1
