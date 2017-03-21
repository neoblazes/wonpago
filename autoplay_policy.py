# Auto play with a model.
#
# Sample usage: python autoplay_policy.py policy_0322_f13 0.3

import numpy as np
import tensorflow as tf

import importlib
import random
import sys

import play_go
import train_lib

if len(sys.argv) < 2:
  print('Usage: python autoplay_policy.py <model_dir> [epsilon]')
  exit(1)
model_dir = sys.argv[1]
epsilon = 0.0
if len(sys.argv) == 3:
  epsilon = float(sys.argv[2])

# Load model and predict
model_fn = importlib.import_module('%s.model_fn' % model_dir)
estimator = model_fn.GetEstimator(model_dir)
board, ko, turn = play_go.InitBoard()
passed = False
while True:
  feature = play_go.ToFeature(board, ko, turn, 0, 0, True, True)
  print(play_go.SPrintBoard(feature[:-1]))
  x_test, _ = train_lib.parse_row(feature, True)  # TODO: make configuable
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
