# Show best move candidates by evaluating all next moves.

import numpy as np
import tensorflow as tf

import sys

import play_go


if len(sys.argv) < 2:
  print('Usage: python eval_policy.py <model_dir>')
  exit(1)
model_dir = sys.argv[1]


# Print out human readable.
def PrintBoard(feature, pred):
  print('%s, predict(W(-1)~B(1)): %f\n' % (play_go.SPrintBoard(feature), pred - 1))


# Load model and predict
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=84)]
regressor = tf.contrib.learn.DNNRegressor(
    model_dir=model_dir,
    feature_columns=feature_columns, hidden_units=[81, 81, 49, 25])

# Main loop
while True:
  feature = input('Enter board feature [[board]*81, last_move, [ko]*2]:\n')
  if isinstance(feature, (list, tuple)):
    feature = list(feature)
  else:
    feature = feature.split(',')
  if len(feature) < 84:
    continue
  feature = list(map(float, feature))[:84]
  board, last_move, ko = play_go.FromFeature(feature)
  next_move = -last_move
  x_test = np.array([feature], dtype=float)
  PrintBoard(feature, list(regressor.predict(x_test))[0])

  # get all features one step forward
  features = play_go.FowardFeatures(feature)
  # Batch eval (for performance)  
  x_test = np.array(features, dtype=float)
  pred_tf = regressor.predict(x_test)
  move_scores = {}
  idx = 0
  for pred in pred_tf:
    move_scores[','.join(list(map(str, features[idx])))] = pred
    idx = idx + 1    
  for k, v in sorted(move_scores.items(), key =lambda x:x[1], reverse = (next_move==1))[:10]:
    PrintBoard(list(map(float, k.split(','))), v)
