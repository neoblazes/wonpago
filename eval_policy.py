# Show best move candidates by evaluating all next moves.

import numpy as np
import tensorflow as tf

import importlib
import random
import sys

import play_go


if len(sys.argv) < 2:
  print('Usage: python eval_policy.py <model_dir> [num_suggestion]')
  print('Set num_suggestion to 0 for autoplay.')
  exit(1)
model_dir = sys.argv[1]
autoplay = False
num_forwards = 4
if len(sys.argv) == 3:
  if sys.argv[2] == '0':
    autoplay = True
  else:
    num_forwards = int(sys.argv[2])

# Load model
model_fn = importlib.import_module('%s.model_fn' % model_dir)
estimator = model_fn.GetEstimator(model_dir)

# Print out human readable.
def PrintBoard(feature, pred):
  print('%s, predict(W(-1)~B(1)): %f\n' % (play_go.SPrintBoard(feature), pred))

# Main loop
feature = 0
forwards = []
while True:
  if autoplay:
    if not feature == 0:
      feature = 1
      while feature < len(forwards) and random.random() < 0.3:
        feature = feature + 1
      # Select random but not evenly.
      print('Selected %dth best move.' % feature)
  else:
    feature = input('0 for new, 1~5 for forward otherwise force feature [[board]*81, last_move, [ko]*2]:\n')
  if isinstance(feature, int) or len(feature) == 1:  # Select from candidates mode.
    feature = int(feature)
    if feature == 0:
      feature = [0] * 81 + [-1] + [0] * 3
    else:
      feature = forwards[int(feature) - 1] + ',0' # TODO: unify result column
  if isinstance(feature, (list, tuple)):
    feature = list(feature)
  else:
    feature = feature.split(',')
  feature = list(map(float, feature))[:-1]  # Remove result column
  board, last_move, ko = play_go.FromFeature(feature[:81] + feature[-3:]) # Uses board only
  x_test = np.array([play_go.ToFeature(board, last_move, ko, 0, True, True)[:-1]], dtype=np.float32)
  PrintBoard(feature, list(estimator.predict(x_test))[0])

  # get all features one step forward
  features = play_go.FowardFeatures(feature)
  # Batch eval (for performance)
  x_test = np.array(features, dtype=np.float32)
  pred_tf = estimator.predict(x_test)
  move_scores = {}
  idx = 0
  for pred in pred_tf:
    move_scores[','.join(list(map(str, features[idx])))] = pred
    idx = idx + 1
  surrender = False
  forwards = []
  for k, v in sorted(move_scores.items(), key =lambda x:x[1], reverse = (last_move==-1))[:num_forwards]:
    forwards.append(k)
    if not autoplay:
      PrintBoard(list(map(float, k.split(','))), v)
    if last_move * v > 1.5:
      surrender = True
  if autoplay:
    if len(forwards) == 0:
      print('No more valid move.')
      break
    if surrender:
      print('Surrender.')
      break
