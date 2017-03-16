# Print out DNN classifier predict for some random openings.
#
# Usage: python opening_predict.py <model_dir> <num_sequence> <num_predict>
# Sample usage: python opening_predict.py tr0310 4 10

import numpy as np
import tensorflow as tf

import importlib
import random
import sys

import play_go

if len(sys.argv) < 4:
  print('Usage: python opening_predict.py <model_dir> <num_sequence> <num_predict>')
  exit(1)
model_dir = sys.argv[1]
num_sequence = int(sys.argv[2])
num_predict = int(sys.argv[3])
print('Test %d sequecne with model %s' % (num_sequence, model_dir))

# Set up kifu
kifus = []
stones = [0] * 81
for i in range(num_sequence):
  stones[i] = (i % 2) * (-2) + 1
tail = [(num_sequence % 2) * 2 - 1, 0, 0]

# Shuffle board and add to kifu list
for _ in range(num_predict):
  random.shuffle(stones)
  board, last_move, ko = play_go.FromFeature(stones + tail)
  kifus.append(play_go.ToFeature(board, last_move, ko, 0, True, True)[:-1])  # remove result column
x_test = np.array(kifus, dtype=np.float32)

# Load model and predict
model_fn = importlib.import_module('%s.model_fn' % model_dir)
estimator = model_fn.GetEstimator(model_dir)
ds_predict_tf  = estimator.predict(x_test)

# Print out human readable.
def PrintBoard(feature, pred):
  print('%s, predict(W(-1)~B(1)): %f\n' % (play_go.SPrintBoard(feature), pred))

# Print out human readable.
BOARD_CHAR = { -1: 'O', 1: '@', 0: '.' }
TURN_MSG = { 1: 'BLACK(@)', -1: 'WHITE(O)' }
idx = 0
for pred in ds_predict_tf:
  PrintBoard(x_test[idx], pred)
  idx = idx + 1
