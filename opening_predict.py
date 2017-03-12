# Print out DNN classifier predict for some random openings.
#
# Usage: python opening_predict.py <model_dir> <num_sequence> <num_predict>
# Sample usage: python opening_predict.py tr0310 4 10

import numpy as np
import tensorflow as tf

import random
import sys

if len(sys.argv) < 4:
  print('Usage: python opening_predict.py <model_dir> <num_sequence> <num_predict>')
  exit(1)
model_dir = sys.argv[1]
num_sequence = int(sys.argv[2])
num_predict = int(sys.argv[3])
print('Test %d sequecne with model %s' % (num_sequence, model_dir))

# Set up kifu
kifus = []
board = [0] * 81
for i in range(num_sequence):
  board[i] = (i % 2) * (-2) + 1
tail = [(num_sequence % 2) * 2 - 1  # mark last_move
        , 0, 0]

# Shuffle board and add to kifu list
for _ in range(num_predict):
  random.shuffle(board)
  kifus.append(board + tail)

x_test = np.array(kifus, dtype=float)
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=84)]

# Load model and predict
regressor = tf.contrib.learn.DNNRegressor(
    model_dir=model_dir,
    feature_columns=feature_columns, hidden_units=[81, 49, 25])
ds_predict_tf  = regressor.predict(x_test)

# Print out human readable.
BOARD_CHAR = { -1: 'O', 1: '@', 0: '.' }
TURN_MSG = { 1: 'BLACK(@)', -1: 'WHITE(O)' }
RESULT_MSG = { 0: 'WHITE', 1: 'JIGO', 2: 'BLACK' }
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
    print outstr
  print('Last move %s, predict(W(-1)~B(1)): %f\n' %
        (TURN_MSG[last_move], pred - 1))
  idx = idx + 1
