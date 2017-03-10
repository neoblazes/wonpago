# Print out DNN classifier predict for Kifu.
# It works for test.csv and produces human readable output.
#
# Usage: python test_predict.py <model_dir> [kifu_feature_csv]
# Sample usage: python test_predict.py 0310 small.csv

import numpy as np
import tensorflow as tf

import sys

if len(sys.argv) < 2:
  pirnt('Usage: python test_predict.py <model_dir> [kifu_feature_csv]')
  exit(1)
model_dir = sys.argv[2]
test_set = sys.argv[3]
print('Test %d file with model in %s' % (test_set, model_dir)

# Load and define features
test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=test_set, target_dtype=np.int, features_dtype=np.float32, target_column=-1)
x_test, y_test = test_set.data, test_set.target
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=84)]

# Load model and predict
classifier = tf.contrib.learn.DNNClassifier(
    model_dir=model_dir, config=config,
    feature_columns=feature_columns, hidden_units=[20, 20], n_classes=3)
ds_predict_tf  = classifier.predict(x_test) 

# Print out human readable.
BOARD_CHAR = { -1: 'O', 1: 'X', 0: '+' }
NEXT_TURN_MSG = { -1: 'BLACK(X)', 1: 'WHITE(O)' }
RESULT_MSG = { 0: 'WHITE', 1: 'JIGO', 2: 'BLACK' }
for i in range(len(ds_predict_tf)):
  feature = x_test[i]
  board = feature[:81]
  last_move = feature[81]
  ko_pos = feature[82:84]
  idx = 0
  for row in range(1:10):
    outstr = ''
    for col in range(1:10):
      if row == ko_pos[0] and col == ko_pos:
        outstr = outstr + '*'
      else:
        outstr = outstr + BOARD_CHAR[board[idx]]
      idx = idx + 1
    print outstr
  print('%s turn, predict: %s, real: %s' %
        (NEXT_TURN_MSG[last_move], RESULT_MSG[ds_predict_tf[i]], RESULT_MSG[y_test[i]]))
