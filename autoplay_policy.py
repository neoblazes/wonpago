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

if len(sys.argv) < 2:
  print('Usage: python autoplay_policy.py <model_dir> [epsilon]')
  exit(1)
model_dir = sys.argv[1]
epsilon = 0.0
if len(sys.argv) == 3:
  epsilon = float(sys.argv[2])

# Extract to lib.
def parse_row(row):
  board_exp = [[0] * 81 for _ in range(3)]
  liberty_idx = 81
  liberty_map = [[0] * 81 for _ in range(2)]
  group_idx = 81 * 2
  group_size = [[0] * 81 for _ in range(2)]
  valid_idx = 81 * 3
  valid_move_exp = [[0] * 81 for _ in range(2)]

  for i in range(81):
    stone = int(row[i])
    # Mark empty, black and white for each layer
    board_exp[stone][i] = 1
    if stone != 0:
      liberty_map[stone - 1][i] = row[liberty_idx + i]
      group_size[stone - 1][i] = row[group_idx + i]
  for i in range(81):
    valid_move = int(row[valid_idx + i])
    valid_move_exp[valid_move - 1][i] = 1
  x_out = []
  for l in [board_exp[0], board_exp[1], board_exp[2], liberty_map[0], liberty_map[1],
            group_size[0], group_size[1], valid_move_exp[0] + valid_move_exp[1]]: # 3 + 2 + 2 + 2 (9 layers)
    x_out += l

  y_num = int(row[-2])
  return x_out, y_num

# Load model and predict
model_fn = importlib.import_module('%s.model_fn' % model_dir)
estimator = model_fn.GetEstimator(model_dir)
board, ko, turn = play_go.InitBoard()
passed = False
while True:
  feature = play_go.ToFeature(board, ko, turn, 0, 0, True, True)
  print(play_go.SPrintBoard(feature[:-1]))
  x_test, _ = parse_row(feature)
  predict = estimator.predict(np.asarray(x_test, dtype=np.float32))
  probabilities = list(predict)[0]['probabilities']

  actions = []
  for i in range(len(probabilities)):
    actions.append([i, probabilities[i]])
  for a in sorted(actions, key = lambda x:x[1], reverse = True):
    if random.random() < epsilon:
      continue
    action = a[0]
    if action == 82:
      print('Surrender signal, but will continue for test')
      continue
    if action == 0:
      if passed:
        print('Both passed')
        exit(1)
      print('Passed')
      passed = True
      break
    valid, ko = play_go.PlayGo(board, turn, play_go.UnpackAction(action))
    if valid:
      passed = False
      break
  turn = play_go.FlipTurn(turn)
